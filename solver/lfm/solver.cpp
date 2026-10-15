module;
#include "cuda.cuh"
module pyro.lfm;
import std;
import pyro.geometry;

namespace pyro::lfm {
    static_assert(sizeof(Config) == sizeof(cuda::lfm::Config));

    Solver::Solver(const Config& settings) : config(settings), parameters(std::bit_cast<cuda::lfm::Config>(settings)) {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            auto layout = [](int3 size) {
                cuda::lfm::Layout result;
                result.size   = size;
                result.tiles  = {(size.x + 7) / 8, (size.y + 7) / 8, (size.z + 7) / 8};
                result.blocks = result.tiles.x * result.tiles.y * result.tiles.z;
                result.count  = result.blocks * cuda::lfm::tileCells;
                return result;
            };
            grid.cells = layout(parameters.resolution);
            allocate(grid.fluid, grid.cells.count);
            history.resize(config.cycleSteps);
            for (int axis = 0; axis < 3; ++axis) {
                int3 size = parameters.resolution;
                if (axis == 0) ++size.x;
                if (axis == 1) ++size.y;
                if (axis == 2) ++size.z;
                grid.faces[axis] = layout(size);
                auto count       = grid.faces[axis].count;
                allocate(grid.boundary[axis], count);
                for (auto field : {&grid.initial, &grid.work, &grid.impulse, &grid.error}) allocate(field->axis[axis], count);
                for (auto& field : history) allocate(field.axis[axis], count);
                for (auto map : {&grid.backward, &grid.forward}) {
                    allocate(map->position[axis], count);
                    allocate(map->tangent[axis], count);
                }
            }
            allocate(grid.history, history.size());
            cuda_check(cudaMemcpyAsync(grid.history, history.data(), history.size() * sizeof(cuda::lfm::Velocity), cudaMemcpyHostToDevice, stream));
            output.resize(std::size_t(config.resolution[0]) * config.resolution[1] * config.resolution[2]);
            allocate(grid.output, output.size());
            pressure.levels = new cuda::lfm::PressureLevel[std::bit_width(static_cast<unsigned>(std::ranges::max(config.resolution)))]{};
            int3 size       = parameters.resolution;
            float weight    = 1;
            for (;;) {
                auto& level         = pressure.levels[pressure.levelCount++];
                level.layout        = layout(size);
                level.regularWeight = weight;
                allocate(level.diagonal, level.layout.count);
                for (auto& edge : level.weight) allocate(edge, level.layout.count);
                allocate(level.solution, level.layout.count);
                allocate(level.temporary, level.layout.count);
                if (pressure.levelCount > 1) allocate(level.rhs, level.layout.count);
                allocate(level.kind, level.layout.blocks);
                if (std::max({size.x, size.y, size.z}) <= 8) break;
                size = {(size.x + 1) / 2, (size.y + 1) / 2, (size.z + 1) / 2};
                weight *= 4;
            }
            auto count = grid.cells.count;
            allocate(pressure.parents, count);
            allocate(pressure.componentCounts, count);
            allocate(pressure.componentSums, count);
            allocate(pressure.blockComponents, grid.cells.blocks);
            allocate(pressure.solution, count);
            allocate(pressure.residual, count);
            allocate(pressure.direction, count);
            allocate(pressure.product, count);
            allocate(pressure.iteration, 1);
            allocate(pressure.reductions, std::size_t(grid.cells.blocks) * 3 + 3);
            statistics.pressureLevels = pressure.levelCount;
            events.resize(2 * config.cycleSteps + 5);
            for (auto& event : events) cuda_check(cudaEventCreate(&event));
            cuda::lfm::initialize(grid, parameters, stream);
            synchronize();
        } catch (...) {
            release_storage();
            throw;
        }
    }
    Solver::~Solver() {
        release_storage();
    }
    int Solver::add_mesh(std::span<const Triangle> triangles) {
        auto mesh = upload_mesh(triangles, 4 * config.cellSize, stream);
        meshes.push_back(mesh);
        statistics.allocatedBytes += triangles.size_bytes() + triangles.size() * sizeof(int) + mesh.leafCount * 2ull * sizeof(cuda::Bounds);
        return static_cast<int>(meshes.size()) - 1;
    }
    void Solver::set_colliders(std::span<const Collider> inputs) {
        if (colliders) {
            cuda_check(cudaFreeAsync(colliders, stream));
            statistics.allocatedBytes -= colliderCount * sizeof(cuda::Collider);
            colliders = nullptr;
        }
        colliderCount = static_cast<int>(inputs.size());
        if (colliderCount) {
            std::vector<cuda::Collider> upload;
            upload.reserve(inputs.size());
            for (const auto& input : inputs) upload.push_back(device_input(input, meshes));
            cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&colliders), upload.size() * sizeof(cuda::Collider), stream));
            statistics.allocatedBytes += upload.size() * sizeof(cuda::Collider);
            cuda_check(cudaMemcpyAsync(colliders, upload.data(), upload.size() * sizeof(cuda::Collider), cudaMemcpyHostToDevice, stream));
            synchronize();
        }
        geometryDirty = true;
    }
    void Solver::advance(double duration) {
        statistics.substeps              = config.cycleSteps;
        statistics.pressureIterations    = 0;
        statistics.advectionMilliseconds = statistics.flowMapMilliseconds = statistics.projectionMilliseconds = 0;
        cuda_check(cudaEventRecord(events[0], stream));
        prepare();
        cuda_check(cudaEventRecord(events[1], stream));
        float dt = static_cast<float>(duration / config.cycleSteps);
        for (int k = 0; k < config.cycleSteps; ++k) {
            auto original  = k == 0 ? grid.initial : k == 1 ? history[0] : history[k - 2];
            auto carrier   = k == 0 ? grid.initial : history[k - 1];
            float interval = k == 0 ? dt * .5f : k == 1 ? dt : 2 * dt;
            cuda::lfm::advect(grid, history[k], original, carrier, interval / config.cellSize, parameters.wind, stream);
            cuda_check(cudaEventRecord(events[2 + 2 * k], stream));
            project(history[k]);
            cuda_check(cudaEventRecord(events[3 + 2 * k], stream));
        }
        int end = 2 * config.cycleSteps;
        cuda::lfm::reconstruct(grid, parameters, dt, stream);
        cuda_check(cudaEventRecord(events[end + 2], stream));
        project(grid.work);
        cuda_check(cudaEventRecord(events[end + 3], stream));
        std::swap(grid.initial, grid.work);
        cuda::lfm::measure(grid, grid.initial, config.cellSize, pressure.reductions, stream);
        double measured[3];
        cuda_check(cudaMemcpyAsync(measured, pressure.reductions + 3 * grid.cells.blocks, sizeof(measured), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaEventRecord(events[end + 4], stream));
        cuda_check(cudaEventSynchronize(events[end + 4]));
        statistics.fluidCells    = static_cast<int>(measured[0]);
        statistics.maxDivergence = static_cast<float>(measured[1]);
        statistics.maxSpeed      = static_cast<float>(measured[2]);
        cuda_check(cudaEventElapsedTime(&statistics.geometryMilliseconds, events[0], events[1]));
        for (int k = 0; k < config.cycleSteps; ++k) {
            float elapsed;
            cuda_check(cudaEventElapsedTime(&elapsed, events[1 + 2 * k], events[2 + 2 * k]));
            statistics.advectionMilliseconds += elapsed;
            cuda_check(cudaEventElapsedTime(&elapsed, events[2 + 2 * k], events[3 + 2 * k]));
            statistics.projectionMilliseconds += elapsed;
        }
        float elapsed;
        cuda_check(cudaEventElapsedTime(&statistics.flowMapMilliseconds, events[end + 1], events[end + 2]));
        cuda_check(cudaEventElapsedTime(&elapsed, events[end + 2], events[end + 3]));
        statistics.projectionMilliseconds += elapsed;
        cuda_check(cudaEventElapsedTime(&statistics.statisticsMilliseconds, events[end + 3], events[end + 4]));
        cuda_check(cudaEventElapsedTime(&statistics.stepMilliseconds, events[0], events[end + 4]));
        time += duration;
    }
    void Solver::synchronize() {
        cuda_check(cudaGetLastError());
        cuda_check(cudaStreamSynchronize(stream));
    }
    void Solver::checkpoint(const char* path, bool restore) {
        prepare();
        std::fstream file(path, std::ios::binary | (restore ? std::ios::in : std::ios::out | std::ios::trunc));
        file.exceptions(std::ios::badbit | std::ios::failbit);
        auto transfer = [&](auto& value) {
            if (restore) file.read(reinterpret_cast<char*>(&value), sizeof(value));
            else file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        std::array<char, 8> magic{'P', 'Y', 'R', 'O', 'L', 'F', 'M', '1'};
        auto expected = magic;
        transfer(magic);
        if (magic != expected) throw std::runtime_error("Unsupported LFM checkpoint");
        transfer(time);
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float> values(grid.faces[axis].count);
            if (restore) {
                file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
                cuda_check(cudaMemcpyAsync(grid.initial.axis[axis], values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                synchronize();
            } else {
                cuda_check(cudaMemcpyAsync(values.data(), grid.initial.axis[axis], values.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                synchronize();
                file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            }
        }
    }
    std::span<const float> Solver::download_vorticity() {
        prepare();
        cuda::lfm::export_vorticity(grid, grid.initial, config.cellSize, stream);
        cuda_check(cudaMemcpyAsync(output.data(), grid.output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        synchronize();
        return output;
    }
    template <typename T>
    void Solver::allocate(T*& pointer, std::size_t count) {
        cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&pointer), count * sizeof(T), stream));
        storage.push_back(pointer);
        statistics.allocatedBytes += count * sizeof(T);
    }
    void Solver::prepare() {
        if (!geometryDirty) return;
        cuda::lfm::geometry(grid, parameters, colliders, colliderCount, stream);
        cuda::lfm::enforce_boundary(grid, grid.initial, parameters.wind, stream);
        cuda::lfm::build_pressure(grid, pressure, stream);
        cuda::lfm::measure(grid, grid.initial, config.cellSize, pressure.reductions, stream);
        double measured[3];
        cuda_check(cudaMemcpyAsync(measured, pressure.reductions + 3 * grid.cells.blocks, sizeof(measured), cudaMemcpyDeviceToHost, stream));
        synchronize();
        statistics.fluidCells    = static_cast<int>(measured[0]);
        statistics.maxDivergence = static_cast<float>(measured[1]);
        statistics.maxSpeed      = static_cast<float>(measured[2]);
        geometryDirty            = false;
    }
    void Solver::project(cuda::lfm::Velocity velocity) {
        cuda::lfm::begin_pressure(grid, velocity, pressure, parameters, stream);
        if (!pressure.graph) {
            cudaGraph_t graph;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < 4; ++i) cuda::lfm::iterate_pressure(pressure, parameters, stream);
            cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&pressure.graph, graph, 0));
            cuda_check(cudaGraphDestroy(graph));
        }
        cuda::IterationState result{};
        for (;;) {
            int remaining = config.pressureMaxIterations - result.iterations;
            if (remaining >= 4) cuda_check(cudaGraphLaunch(pressure.graph, stream));
            else
                for (int i = 0; i < remaining; ++i) cuda::lfm::iterate_pressure(pressure, parameters, stream);
            cuda_check(cudaMemcpyAsync(&result, pressure.iteration, sizeof(result), cudaMemcpyDeviceToHost, stream));
            synchronize();
            if (result.converged) {
                cuda::lfm::check_pressure(grid, velocity, pressure, parameters, stream);
                cuda_check(cudaMemcpyAsync(&result, pressure.iteration, sizeof(result), cudaMemcpyDeviceToHost, stream));
                synchronize();
                if (result.converged) break;
            }
            if (result.failed || result.iterations >= config.pressureMaxIterations) throw std::runtime_error("LFM pressure did not converge: iterations=" + std::to_string(result.iterations) + ", residual=" + std::to_string(result.maxResidual / config.cellSize));
        }
        statistics.pressureIterations += result.iterations;
        cuda::lfm::apply_pressure(grid, velocity, pressure, stream);
    }
    void Solver::release_storage() {
        if (stream) cudaStreamSynchronize(stream);
        if (pressure.graph) cudaGraphExecDestroy(pressure.graph);
        for (auto event : events)
            if (event) cudaEventDestroy(event);
        if (colliders) cudaFreeAsync(colliders, stream);
        for (auto mesh : meshes) {
            cudaFreeAsync(mesh.triangles, stream);
            cudaFreeAsync(mesh.nodes, stream);
            cudaFreeAsync(mesh.order, stream);
        }
        for (auto pointer : storage) cudaFreeAsync(pointer, stream);
        delete[] pressure.levels;
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
} // namespace pyro::lfm
