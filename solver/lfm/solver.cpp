module;
#include "cuda.cuh"
module pyro.lfm;
import std;
import pyro.geometry;

namespace pyro::lfm {
    static_assert(sizeof(Config) == sizeof(cuda::lfm::Config));

    Solver::Solver(const Config& settings) : config(settings), parameters(std::bit_cast<cuda::lfm::Config>(settings)) {
        if (auto cuda_result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        try {
            auto layout = [](int3 size) {
                cuda::lfm::Layout result;
                result.size   = size;
                result.tiles  = {(size.x + 7) / 8, (size.y + 7) / 8, (size.z + 7) / 8};
                result.blocks = result.tiles.x * result.tiles.y * result.tiles.z;
                result.count  = result.blocks * cuda::lfm::tile_cells;
                return result;
            };
            grid.cells = layout(parameters.resolution);
            allocate(grid.fluid, grid.cells.count);
            history.resize(config.cycle_steps);
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
            if (auto cuda_result = cudaMemcpyAsync(grid.history, history.data(), history.size() * sizeof(cuda::lfm::Velocity), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            output.resize(std::size_t(config.resolution[0]) * config.resolution[1] * config.resolution[2]);
            allocate(grid.output, output.size());
            pressure.levels = new cuda::lfm::PressureLevel[std::bit_width(static_cast<unsigned>(std::ranges::max(config.resolution)))]{};
            int3 size       = parameters.resolution;
            float weight    = 1;
            for (;;) {
                auto& level          = pressure.levels[pressure.level_count++];
                level.layout         = layout(size);
                level.regular_weight = weight;
                allocate(level.diagonal, level.layout.count);
                for (auto& edge : level.weight) allocate(edge, level.layout.count);
                allocate(level.solution, level.layout.count);
                allocate(level.temporary, level.layout.count);
                if (pressure.level_count > 1) allocate(level.rhs, level.layout.count);
                allocate(level.kind, level.layout.blocks);
                if (std::max({size.x, size.y, size.z}) <= 8) break;
                size = {(size.x + 1) / 2, (size.y + 1) / 2, (size.z + 1) / 2};
                weight *= 4;
            }
            auto count = grid.cells.count;
            allocate(pressure.parents, count);
            allocate(pressure.component_counts, count);
            allocate(pressure.component_sums, count);
            allocate(pressure.block_components, grid.cells.blocks);
            allocate(pressure.solution, count);
            allocate(pressure.residual, count);
            allocate(pressure.direction, count);
            allocate(pressure.product, count);
            allocate(pressure.iteration, 1);
            allocate(pressure.reductions, std::size_t(grid.cells.blocks) * 3 + 3);
            statistics.pressure_levels = pressure.level_count;
            events.resize(2 * config.cycle_steps + 5);
            for (auto& event : events)
                if (auto cuda_result = cudaEventCreate(&event); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
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
        auto mesh = upload_mesh(triangles, 4 * config.cell_size, stream);
        meshes.push_back(mesh);
        statistics.allocated_bytes += triangles.size_bytes() + triangles.size() * sizeof(int) + mesh.leaf_count * 2ull * sizeof(cuda::Bounds);
        return static_cast<int>(meshes.size()) - 1;
    }
    void Solver::set_colliders(std::span<const Collider> inputs) {
        if (colliders) {
            if (auto cuda_result = cudaFreeAsync(colliders, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            statistics.allocated_bytes -= collider_count * sizeof(cuda::Collider);
            colliders = nullptr;
        }
        collider_count = static_cast<int>(inputs.size());
        if (collider_count) {
            std::vector<cuda::Collider> upload;
            upload.reserve(inputs.size());
            for (const auto& input : inputs) upload.push_back({std::bit_cast<float3>(input.center), std::bit_cast<float3>(input.half_extent), std::bit_cast<float4>(input.orientation), std::bit_cast<float3>(input.linear_velocity), std::bit_cast<float3>(input.angular_velocity), input.pose_time, static_cast<cuda::Shape>(input.shape), input.shape == Shape::mesh ? meshes[input.mesh] : cuda::Mesh{}, std::bit_cast<cuda::Motion>(input.motion), input.surface_offset});
            if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&colliders), upload.size() * sizeof(cuda::Collider), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            statistics.allocated_bytes += upload.size() * sizeof(cuda::Collider);
            if (auto cuda_result = cudaMemcpyAsync(colliders, upload.data(), upload.size() * sizeof(cuda::Collider), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            synchronize();
        }
        geometry_dirty = true;
    }
    void Solver::advance(double duration) {
        statistics.substeps               = config.cycle_steps;
        statistics.pressure_iterations    = 0;
        statistics.advection_milliseconds = statistics.flow_map_milliseconds = statistics.projection_milliseconds = 0;
        if (auto cuda_result = cudaEventRecord(events[0], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        prepare();
        if (auto cuda_result = cudaEventRecord(events[1], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        float dt = static_cast<float>(duration / config.cycle_steps);
        for (int k = 0; k < config.cycle_steps; ++k) {
            auto original  = k == 0 ? grid.initial : k == 1 ? history[0] : history[k - 2];
            auto carrier   = k == 0 ? grid.initial : history[k - 1];
            float interval = k == 0 ? dt * .5f : k == 1 ? dt : 2 * dt;
            cuda::lfm::advect(grid, history[k], original, carrier, interval / config.cell_size, parameters.wind, stream);
            if (auto cuda_result = cudaEventRecord(events[2 + 2 * k], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            project(history[k]);
            if (auto cuda_result = cudaEventRecord(events[3 + 2 * k], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        }
        int end = 2 * config.cycle_steps;
        cuda::lfm::reconstruct(grid, parameters, dt, stream);
        if (auto cuda_result = cudaEventRecord(events[end + 2], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        project(grid.work);
        if (auto cuda_result = cudaEventRecord(events[end + 3], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        std::swap(grid.initial, grid.work);
        cuda::lfm::measure(grid, grid.initial, config.cell_size, pressure.reductions, stream);
        double measured[3];
        if (auto cuda_result = cudaMemcpyAsync(measured, pressure.reductions + 3 * grid.cells.blocks, sizeof(measured), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaEventRecord(events[end + 4], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaEventSynchronize(events[end + 4]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        statistics.fluid_cells    = static_cast<int>(measured[0]);
        statistics.max_divergence = static_cast<float>(measured[1]);
        statistics.max_speed      = static_cast<float>(measured[2]);
        if (auto cuda_result = cudaEventElapsedTime(&statistics.geometry_milliseconds, events[0], events[1]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        for (int k = 0; k < config.cycle_steps; ++k) {
            float elapsed;
            if (auto cuda_result = cudaEventElapsedTime(&elapsed, events[1 + 2 * k], events[2 + 2 * k]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            statistics.advection_milliseconds += elapsed;
            if (auto cuda_result = cudaEventElapsedTime(&elapsed, events[2 + 2 * k], events[3 + 2 * k]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            statistics.projection_milliseconds += elapsed;
        }
        float elapsed;
        if (auto cuda_result = cudaEventElapsedTime(&statistics.flow_map_milliseconds, events[end + 1], events[end + 2]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaEventElapsedTime(&elapsed, events[end + 2], events[end + 3]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        statistics.projection_milliseconds += elapsed;
        if (auto cuda_result = cudaEventElapsedTime(&statistics.statistics_milliseconds, events[end + 3], events[end + 4]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaEventElapsedTime(&statistics.step_milliseconds, events[0], events[end + 4]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        time += duration;
    }
    void Solver::synchronize() {
        if (auto cuda_result = cudaGetLastError(); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
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
                if (auto cuda_result = cudaMemcpyAsync(grid.initial.axis[axis], values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                synchronize();
            } else {
                if (auto cuda_result = cudaMemcpyAsync(values.data(), grid.initial.axis[axis], values.size() * sizeof(float), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                synchronize();
                file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            }
        }
    }
    std::span<const float> Solver::download_vorticity() {
        prepare();
        cuda::lfm::export_vorticity(grid, grid.initial, config.cell_size, stream);
        if (auto cuda_result = cudaMemcpyAsync(output.data(), grid.output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        synchronize();
        return output;
    }
    template <typename T>
    void Solver::allocate(T*& pointer, std::size_t count) {
        if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&pointer), count * sizeof(T), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        storage.push_back(pointer);
        statistics.allocated_bytes += count * sizeof(T);
    }
    void Solver::prepare() {
        if (!geometry_dirty) return;
        cuda::lfm::geometry(grid, parameters, colliders, collider_count, stream);
        cuda::lfm::enforce_boundary(grid, grid.initial, parameters.wind, stream);
        cuda::lfm::build_pressure(grid, pressure, stream);
        cuda::lfm::measure(grid, grid.initial, config.cell_size, pressure.reductions, stream);
        double measured[3];
        if (auto cuda_result = cudaMemcpyAsync(measured, pressure.reductions + 3 * grid.cells.blocks, sizeof(measured), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        synchronize();
        statistics.fluid_cells    = static_cast<int>(measured[0]);
        statistics.max_divergence = static_cast<float>(measured[1]);
        statistics.max_speed      = static_cast<float>(measured[2]);
        geometry_dirty            = false;
    }
    void Solver::project(cuda::lfm::Velocity velocity) {
        cuda::lfm::begin_pressure(grid, velocity, pressure, parameters, stream);
        if (!pressure.graph) {
            cudaGraph_t graph;
            if (auto cuda_result = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            for (int i = 0; i < 4; ++i) cuda::lfm::iterate_pressure(pressure, parameters, stream);
            if (auto cuda_result = cudaStreamEndCapture(stream, &graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaGraphInstantiate(&pressure.graph, graph, 0); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaGraphDestroy(graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        }
        cuda::lfm::IterationState result{};
        for (;;) {
            int remaining = config.pressure_max_iterations - result.iterations;
            if (remaining < 4)
                for (int i = 0; i < remaining; ++i) cuda::lfm::iterate_pressure(pressure, parameters, stream);
            else if (auto cuda_result = cudaGraphLaunch(pressure.graph, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMemcpyAsync(&result, pressure.iteration, sizeof(result), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            synchronize();
            if (result.converged) {
                cuda::lfm::check_pressure(grid, velocity, pressure, parameters, stream);
                if (auto cuda_result = cudaMemcpyAsync(&result, pressure.iteration, sizeof(result), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                synchronize();
                if (result.converged) break;
            }
            if (result.failed || result.iterations >= config.pressure_max_iterations) throw std::runtime_error("LFM pressure did not converge: iterations=" + std::to_string(result.iterations) + ", residual=" + std::to_string(result.max_residual / config.cell_size));
        }
        statistics.pressure_iterations += result.iterations;
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
