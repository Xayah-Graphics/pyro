module;
#include "cuda.cuh"

module pyro.sparse;
import std;

namespace pyro::sparse {
    // Host arrays and native CUDA vectors preserve the binary transfer layout.
    static_assert(sizeof(std::array<float, 3>) == sizeof(float3) && alignof(std::array<float, 3>) == alignof(float3));
    static_assert(sizeof(std::array<int, 3>) == sizeof(int3) && alignof(std::array<int, 3>) == alignof(int3));
    static_assert(sizeof(Config) == sizeof(cuda::Config) && alignof(Config) == alignof(cuda::Config));
    static_assert(sizeof(Motion) == sizeof(cuda::Motion) && alignof(Motion) == alignof(cuda::Motion));
    static_assert(sizeof(Triangle) == sizeof(cuda::Triangle) && alignof(Triangle) == alignof(cuda::Triangle));
    static_assert(sizeof(DensityBlock) == sizeof(cuda::DensityBlock) && alignof(DensityBlock) == alignof(cuda::DensityBlock));
    static_assert(sizeof(GridBlock) == sizeof(cuda::GridBlock) && alignof(GridBlock) == alignof(cuda::GridBlock));

    static void cuda_check(cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    Solver::Solver(const Config& settings) : config(settings), parameters(std::bit_cast<cuda::Config>(settings)) {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaMemPool_t pool;
        cuda_check(cudaDeviceGetDefaultMemPool(&pool, 0));
        unsigned long long threshold = ~0ull;
        cuda_check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
        try {
            allocate_layout(grid.layout, config.maxBlocks);
            auto cells = static_cast<unsigned long long>(config.maxBlocks) * cuda::blockCells;
            allocate(grid.nextKeys, grid.layout.tableCapacity);
            allocate(grid.nextIndices, grid.layout.tableCapacity);
            allocate(grid.requested, grid.layout.tableCapacity);
            allocate(grid.roles, config.maxBlocks);
            allocate(grid.freeBlocks, config.maxBlocks);
            allocate(grid.freeCount, 1);
            allocate(grid.age, config.maxBlocks);
            allocate(grid.fluid, cells);
            allocate(grid.boundaryDirty, config.maxBlocks);
            allocate(grid.solidDivergence, cells);
            for (auto& phi : grid.distance) allocate(phi, cells);
            for (auto& a : grid.aperture) allocate(a, cells);
            for (auto& u : grid.solidVelocity) allocate(u, cells);
            cuda::Fields* fields[]{&grid.state, &grid.midpoint, &grid.next, &grid.transported};
            for (auto field : fields) {
                for (auto& u : field->velocity) allocate(u, cells);
                allocate(field->smoke, cells);
                allocate(field->temperature, cells);
            }
            pressure.levels      = new cuda::PressureLevel[config.maxPressureLevels]{};
            auto& fine           = pressure.levels[0];
            fine.storageCapacity = config.maxBlocks;
            allocate(fine.diagonal, cells);
            allocate(fine.boundary, cells);
            for (auto& a : fine.face) allocate(a, cells);
            allocate(fine.solution, cells);
            allocate(fine.rhs, cells);
            allocate(fine.temporary, cells);
            allocate(pressure.direction, cells);
            allocate(pressure.product, cells);
            allocate(pressure.solution, cells);
            allocate(pressure.residual, cells);
            allocate(pressure.parents, cells);
            allocate(pressure.openComponents, cells);
            allocate(pressure.anchors, cells);
            allocate(pressure.reductions, config.maxBlocks * 6ull + 6);
            allocate(pressure.iteration, 1);
            allocate(error, 1);
            allocate(maximum, 2);
            cuda_check(cudaStreamSynchronize(stream));
        } catch (...) {
            this->~Solver();
            throw;
        }
    }

    Solver::~Solver() {
        // cudaFreeAsync is ordered after outstanding work; teardown must not throw.
        auto free = [stream = stream](void* pointer) {
            if (pointer) cudaFreeAsync(pointer, stream);
        };
        free(grid.layout.keys);
        free(grid.layout.indices);
        free(grid.layout.coordinates);
        free(grid.layout.active);
        free(grid.layout.neighbors);
        free(grid.layout.count);
        free(grid.nextKeys);
        free(grid.nextIndices);
        free(grid.requested);
        free(grid.roles);
        free(grid.freeBlocks);
        free(grid.freeCount);
        free(grid.age);
        free(grid.fluid);
        free(grid.boundaryDirty);
        free(grid.solidDivergence);
        for (auto phi : grid.distance) free(phi);
        for (auto a : grid.aperture) free(a);
        for (auto u : grid.solidVelocity) free(u);
        cuda::Fields fields[]{grid.state, grid.midpoint, grid.next, grid.transported};
        for (auto field : fields) {
            for (auto u : field.velocity) free(u);
            free(field.smoke);
            free(field.temperature);
        }
        if (pressure.levels)
            for (int l = 0; l < config.maxPressureLevels; ++l) {
                auto level = pressure.levels[l];
                if (l) {
                    free(level.layout.keys);
                    free(level.layout.indices);
                    free(level.layout.coordinates);
                    free(level.layout.active);
                    free(level.layout.neighbors);
                    free(level.layout.count);
                }
                free(level.diagonal);
                free(level.boundary);
                for (auto face : level.face) free(face);
                free(level.solution);
                free(level.rhs);
                free(level.temporary);
            }
        delete[] pressure.levels;
        free(pressure.direction);
        free(pressure.product);
        free(pressure.solution);
        free(pressure.residual);
        free(pressure.parents);
        free(pressure.openComponents);
        free(pressure.anchors);
        free(pressure.reductions);
        free(pressure.iteration);
        if (pressure.graph) cudaGraphExecDestroy(pressure.graph);
        free(sources);
        free(colliders);
        for (auto mesh : meshes) {
            free(mesh.triangles);
            free(mesh.nodes);
            free(mesh.order);
        }
        free(error);
        free(maximum);
        free(densityDevice);
        free(densityCount);
        free(gridDevice);
        cudaStreamSynchronize(stream);
        if (densityHost) cudaFreeHost(densityHost);
        if (gridHost) cudaFreeHost(gridHost);
        cudaStreamDestroy(stream);
    }

    void Solver::set_sources(const Source* data, int count) {
        if (count > sourceCapacity) {
            release(sources, sourceCapacity);
            allocate(sources, count);
            sourceCapacity = count;
        }
        sourceCount = count;
        if (!count) return;
        std::vector<cuda::Source> upload(count);
        for (int i = 0; i < count; ++i) {
            const auto& source     = data[i];
            auto& target           = upload[i];
            target.center          = std::bit_cast<float3>(source.center);
            target.halfExtent      = std::bit_cast<float3>(source.halfExtent);
            target.velocity        = std::bit_cast<float3>(source.velocity);
            target.smokeRate       = source.smokeRate;
            target.temperatureRate = source.temperatureRate;
            target.velocityRate    = source.velocityRate;
            target.startTime       = source.startTime;
            target.endTime         = source.endTime;
            target.shape           = static_cast<cuda::Shape>(source.shape);
            target.mesh            = source.shape == Shape::mesh ? meshes[source.mesh] : cuda::Mesh{};
            target.motion          = std::bit_cast<cuda::Motion>(source.motion);
            target.inheritVelocity = source.inheritVelocity;
        }
        cuda_check(cudaMemcpyAsync(sources, upload.data(), count * sizeof(cuda::Source), cudaMemcpyHostToDevice, stream));
        synchronize();
    }

    void Solver::set_colliders(const Collider* data, int count) {
        geometryDirty       = geometryDirty || !grid.staticGeometry || colliderInputs.size() != static_cast<std::size_t>(count);
        grid.staticGeometry = true;
        for (int i = 0; i < count; ++i) {
            const auto& current = data[i];
            if (current.shape == Shape::mesh) {
                for (float derivative : current.motion.derivative) grid.staticGeometry &= derivative == 0.0f;
            } else {
                for (float velocity : current.linearVelocity) grid.staticGeometry &= velocity == 0.0f;
                for (float velocity : current.angularVelocity) grid.staticGeometry &= velocity == 0.0f;
            }
            if (!geometryDirty) {
                const auto& previous = colliderInputs[i];
                geometryDirty        = current.shape != previous.shape || current.mesh != previous.mesh || current.surfaceOffset != previous.surfaceOffset;
                if (current.shape == Shape::mesh) geometryDirty |= !std::ranges::equal(current.motion.matrix, previous.motion.matrix);
                else geometryDirty |= current.center != previous.center || current.halfExtent != previous.halfExtent || current.orientation != previous.orientation;
            }
        }
        geometryDirty |= !grid.staticGeometry;
        colliderInputs.assign_range(std::span(data, static_cast<std::size_t>(count)));
        if (count > colliderCapacity) {
            release(colliders, colliderCapacity);
            allocate(colliders, count);
            colliderCapacity = count;
        }
        colliderCount = count;
        if (!count) return;
        std::vector<cuda::Collider> upload(count);
        for (int i = 0; i < count; ++i) {
            const auto& collider   = data[i];
            auto& target           = upload[i];
            target.center          = std::bit_cast<float3>(collider.center);
            target.halfExtent      = std::bit_cast<float3>(collider.halfExtent);
            target.orientation     = std::bit_cast<float4>(collider.orientation);
            target.linearVelocity  = std::bit_cast<float3>(collider.linearVelocity);
            target.angularVelocity = std::bit_cast<float3>(collider.angularVelocity);
            target.poseTime        = collider.poseTime;
            target.shape           = static_cast<cuda::Shape>(collider.shape);
            target.mesh            = collider.shape == Shape::mesh ? meshes[collider.mesh] : cuda::Mesh{};
            target.motion          = std::bit_cast<cuda::Motion>(collider.motion);
            target.surfaceOffset   = collider.surfaceOffset;
        }
        cuda_check(cudaMemcpyAsync(colliders, upload.data(), count * sizeof(cuda::Collider), cudaMemcpyHostToDevice, stream));
        synchronize();
    }

    int Solver::add_mesh(std::span<const Triangle> triangles) {
        cuda::Mesh mesh;
        mesh.triangleCount = static_cast<int>(triangles.size());
        mesh.leafCount     = static_cast<int>(std::bit_ceil(triangles.size()));
        mesh.band          = config.cellSize * 4.0f;
        mesh.bounds        = {{1.0e30f, 1.0e30f, 1.0e30f}, {-1.0e30f, -1.0e30f, -1.0e30f}};
        for (auto t : triangles)
            for (auto p : {t.a, t.b, t.c}) {
                mesh.bounds.lower = {std::min(mesh.bounds.lower.x, p[0]), std::min(mesh.bounds.lower.y, p[1]), std::min(mesh.bounds.lower.z, p[2])};
                mesh.bounds.upper = {std::max(mesh.bounds.upper.x, p[0]), std::max(mesh.bounds.upper.y, p[1]), std::max(mesh.bounds.upper.z, p[2])};
            }
        allocate(mesh.triangles, mesh.triangleCount);
        allocate(mesh.order, mesh.triangleCount);
        allocate(mesh.nodes, mesh.leafCount * 2);
        cuda_check(cudaMemcpyAsync(mesh.triangles, triangles.data(), triangles.size_bytes(), cudaMemcpyHostToDevice, stream));
        cuda::build_mesh(mesh, stream);
        synchronize();
        meshes.push_back(mesh);
        return static_cast<int>(meshes.size()) - 1;
    }

    void Solver::advance(double duration) {
        auto begin                = std::chrono::steady_clock::now();
        auto bytes                = statistics.allocatedBytes;
        statistics                = {};
        statistics.allocatedBytes = bytes;
        double endTime            = time + duration;
        while (time < endTime) {
            if (!grid.staticGeometry || geometryDirty) {
                cuda::sample_geometry(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time), 0, false, stream);
                cuda::update_boundary(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time), 0, false, stream);
                geometryDirty = false;
            }
            cuda::measure(grid, grid.state, parameters, sources, sourceCount, colliders, colliderCount, static_cast<float>(time), pressure.reductions, maximum, false, stream);
            float bounds[2];
            cuda_check(cudaMemcpyAsync(bounds, maximum, sizeof(bounds), cudaMemcpyDeviceToHost, stream));
            synchronize();
            // Stable positive root of v*dt + a*dt^2 <= CFL*h, including source acceleration.
            float distance     = config.cfl * config.cellSize;
            float safeStep     = 2.0f * distance / (bounds[0] + std::sqrt(bounds[0] * bounds[0] + 4.0f * bounds[1] * distance) + 1.0e-20f);
            float dt           = static_cast<float>(std::min(endTime - time, static_cast<double>(std::min(config.maxTimeStep, safeStep))));
            auto topologyBegin = std::chrono::steady_clock::now();
            cuda::request_topology(grid, parameters, sources, sourceCount, colliders, colliderCount, static_cast<float>(time), dt, error, stream);
            std::swap(grid.layout.keys, grid.nextKeys);
            std::swap(grid.layout.indices, grid.nextIndices);
            int deviceError;
            cuda_check(cudaMemcpyAsync(&grid.layout.blocks, grid.layout.count, sizeof(int), cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaMemcpyAsync(&deviceError, error, sizeof(int), cudaMemcpyDeviceToHost, stream));
            synchronize();
            if (deviceError) throw std::runtime_error("Sparse block pool exhausted; increase Config::maxBlocks.");
            cuda::initialize_topology(grid, parameters.wind, stream);
            hierarchyDirty        = true;
            grid.geometryTime     = static_cast<float>(time);
            grid.geometryDuration = dt;
            int samples           = grid.staticGeometry ? 1 : 3;
            for (int sample = 0; sample < samples; ++sample) cuda::sample_geometry(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time) + dt * (0.5f * sample), sample, grid.staticGeometry, stream);
            cuda::update_boundary(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time), 0, grid.staticGeometry, stream);
            synchronize();
            statistics.topologyMilliseconds += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - topologyBegin).count();
            if (grid.layout.blocks) {
                cuda::advect(grid, parameters, grid.midpoint, grid.state, grid.state, colliders, colliderCount, static_cast<float>(time), dt * 0.5f, stream);
                cuda::apply_sources_forces(grid, parameters, grid.midpoint, grid.state, sources, sourceCount, static_cast<float>(time), dt * 0.5f, stream);
                if (!grid.staticGeometry) cuda::update_boundary(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time) + dt * 0.5f, 1, false, stream);
                project(grid.midpoint);
                cuda::advect(grid, parameters, grid.next, grid.state, grid.midpoint, colliders, colliderCount, static_cast<float>(time), dt, stream);
                cuda::apply_sources_forces(grid, parameters, grid.next, grid.midpoint, sources, sourceCount, static_cast<float>(time), dt, stream);
                if (!grid.staticGeometry) cuda::update_boundary(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time) + dt, 2, false, stream);
                project(grid.next);
                std::swap(grid.state, grid.next);
            }
            time += dt;
            ++statistics.substeps;
        }
        cuda::measure(grid, grid.state, parameters, sources, sourceCount, colliders, colliderCount, static_cast<float>(time), pressure.reductions, maximum, true, stream);
        double result[6];
        cuda_check(cudaMemcpyAsync(result, pressure.reductions + config.maxBlocks * 6ull, sizeof(result), cudaMemcpyDeviceToHost, stream));
        synchronize();
        statistics.blocks              = grid.layout.blocks;
        statistics.fluidCells          = static_cast<int>(result[0]);
        statistics.maxDivergence       = static_cast<float>(result[1]);
        statistics.maxSpeed            = static_cast<float>(result[2]);
        statistics.smokeIntegral       = result[3];
        statistics.temperatureIntegral = result[4];
        statistics.stepMilliseconds    = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (!std::isfinite(statistics.maxDivergence) || statistics.maxDivergence > config.divergenceTolerance) throw std::runtime_error("Projection divergence " + std::to_string(statistics.maxDivergence) + " exceeds " + std::to_string(config.divergenceTolerance));
    }

    void Solver::synchronize() {
        cuda_check(cudaGetLastError());
        cuda_check(cudaStreamSynchronize(stream));
    }

    void Solver::checkpoint(const char* path, bool restore) {
        synchronize();
        std::fstream file(path, std::ios::binary | (restore ? std::ios::in : std::ios::out | std::ios::trunc));
        file.exceptions(std::ios::badbit | std::ios::failbit);
        auto scalar = [&]<typename T>(T& value) {
            if (restore) file.read(reinterpret_cast<char*>(&value), sizeof(T));
            else file.write(reinterpret_cast<const char*>(&value), sizeof(T));
        };
        auto deviceArray = [&]<typename T>(T* device, std::size_t count) {
            std::vector<T> host(count);
            if (restore) {
                file.read(reinterpret_cast<char*>(host.data()), count * sizeof(T));
                if (count) cuda_check(cudaMemcpyAsync(device, host.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream));
            } else {
                if (count) cuda_check(cudaMemcpyAsync(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
                synchronize();
                file.write(reinterpret_cast<const char*>(host.data()), count * sizeof(T));
            }
            synchronize();
        };
        unsigned long long format = 0x32305453524f5950ull;
        scalar(format);
        scalar(time);
        scalar(grid.layout.blocks);
        int freeCount = 0;
        if (!restore && grid.layout.blocks) cuda_check(cudaMemcpyAsync(&freeCount, grid.freeCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        synchronize();
        scalar(freeCount);
        deviceArray(grid.freeBlocks, freeCount);
        deviceArray(grid.layout.keys, grid.layout.tableCapacity);
        deviceArray(grid.layout.indices, grid.layout.tableCapacity);
        cuda::StateBlock* packed = nullptr;
        if (grid.layout.blocks) allocate(packed, grid.layout.blocks);
        try {
            if (!restore) cuda::transfer_state(grid, packed, false, stream);
            deviceArray(packed, grid.layout.blocks);
            if (restore) {
                cuda::transfer_state(grid, packed, true, stream);
                cuda_check(cudaMemcpyAsync(grid.layout.count, &grid.layout.blocks, sizeof(int), cudaMemcpyHostToDevice, stream));
                cuda_check(cudaMemcpyAsync(grid.freeCount, &freeCount, sizeof(int), cudaMemcpyHostToDevice, stream));
                cuda::build_neighbors(grid.layout, stream);
                hierarchyDirty = true;
                geometryDirty  = true;
            }
            synchronize();
        } catch (...) {
            release(packed, grid.layout.blocks);
            throw;
        }
        release(packed, grid.layout.blocks);
    }

    std::span<const DensityBlock> Solver::download_density() {
        if (!densityDevice) {
            allocate(densityDevice, config.maxBlocks);
            allocate(densityCount, 1);
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&densityHost), config.maxBlocks * sizeof(DensityBlock)));
        }
        cuda::pack_density(grid, densityDevice, densityCount, stream);
        int count;
        cuda_check(cudaMemcpyAsync(&count, densityCount, sizeof(int), cudaMemcpyDeviceToHost, stream));
        synchronize();
        if (count) cuda_check(cudaMemcpyAsync(densityHost, densityDevice, count * sizeof(DensityBlock), cudaMemcpyDeviceToHost, stream));
        synchronize();
        return {densityHost, static_cast<std::size_t>(count)};
    }

    std::span<const GridBlock> Solver::download_grid() {
        if (!gridDevice) {
            allocate(gridDevice, config.maxBlocks);
            cuda_check(cudaMallocHost(reinterpret_cast<void**>(&gridHost), config.maxBlocks * sizeof(GridBlock)));
        }
        cuda::pack_grid(grid, gridDevice, stream);
        if (grid.layout.blocks) cuda_check(cudaMemcpyAsync(gridHost, gridDevice, grid.layout.blocks * sizeof(GridBlock), cudaMemcpyDeviceToHost, stream));
        synchronize();
        return {gridHost, static_cast<std::size_t>(grid.layout.blocks)};
    }

    template <typename T>
    void Solver::allocate(T*& pointer, unsigned long long count) {
        cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&pointer), count * sizeof(T), stream));
        statistics.allocatedBytes += count * sizeof(T);
    }

    template <typename T>
    void Solver::release(T*& pointer, unsigned long long count) {
        if (!pointer) return;
        cuda_check(cudaFreeAsync(pointer, stream));
        pointer = nullptr;
        statistics.allocatedBytes -= count * sizeof(T);
    }

    void Solver::allocate_layout(cuda::Layout& layout, int capacity) {
        layout.capacity      = capacity;
        layout.tableCapacity = 1;
        while (layout.tableCapacity < capacity * 4) layout.tableCapacity <<= 1;
        allocate(layout.keys, layout.tableCapacity);
        allocate(layout.indices, layout.tableCapacity);
        allocate(layout.coordinates, capacity);
        allocate(layout.active, capacity);
        allocate(layout.neighbors, capacity * 6);
        allocate(layout.count, 1);
        cuda_check(cudaMemsetAsync(layout.keys, 255, layout.tableCapacity * sizeof(unsigned long long), stream));
        cuda_check(cudaMemsetAsync(layout.indices, 255, layout.tableCapacity * sizeof(int), stream));
        cuda_check(cudaMemsetAsync(layout.count, 0, sizeof(int), stream));
    }

    void Solver::free_layout(cuda::Layout& layout) {
        release(layout.keys, layout.tableCapacity);
        release(layout.indices, layout.tableCapacity);
        release(layout.coordinates, layout.capacity);
        release(layout.active, layout.capacity);
        release(layout.neighbors, layout.capacity * 6);
        release(layout.count, 1);
        layout = {};
    }

    void Solver::project(cuda::Fields fields) {
        auto& workspace = pressure;
        auto& fine      = workspace.levels[0];
        fine.layout     = grid.layout;
        if (!fine.layout.blocks) return;
        auto begin = std::chrono::steady_clock::now();
        // Only the RHS changes between the two projections with stationary geometry.
        bool rebuild = hierarchyDirty || !grid.staticGeometry;
        cuda::assemble_pressure(grid, fields, workspace, config.cellSize, colliderCount != 0, rebuild, stream);
        workspace.levelCount = 1;
        for (int l = 1; l < config.maxPressureLevels; ++l) {
            auto& previous = workspace.levels[l - 1];
            auto& level    = workspace.levels[l];
            if (hierarchyDirty && level.layout.capacity < previous.layout.blocks) {
                free_layout(level.layout);
                allocate_layout(level.layout, previous.layout.blocks);
            }
            if (hierarchyDirty) {
                cuda::coarse_topology(previous.layout, level.layout, stream);
                cuda_check(cudaMemcpyAsync(&level.layout.blocks, level.layout.count, sizeof(int), cudaMemcpyDeviceToHost, stream));
                cuda_check(cudaStreamSynchronize(stream));
            }
            if (level.storageCapacity < level.layout.blocks) {
                auto old = static_cast<unsigned long long>(level.storageCapacity) * cuda::blockCells;
                release(level.diagonal, old);
                release(level.boundary, old);
                for (auto& face : level.face) release(face, old);
                release(level.solution, old);
                release(level.rhs, old);
                release(level.temporary, old);
                level.storageCapacity = level.layout.blocks + level.layout.blocks / 4;
                auto size             = static_cast<unsigned long long>(level.storageCapacity) * cuda::blockCells;
                allocate(level.diagonal, size);
                allocate(level.boundary, size);
                for (auto& face : level.face) allocate(face, size);
                allocate(level.solution, size);
                allocate(level.rhs, size);
                allocate(level.temporary, size);
            }
            if (rebuild) cuda::coarse_matrix(previous, level, stream);
            ++workspace.levelCount;
            if (l >= 6 && level.layout.blocks <= 8) break;
        }
        cuda_check(cudaMemsetAsync(workspace.iteration, 0, sizeof(cuda::IterationState), stream));
        cuda::begin_pressure(workspace, parameters, stream);
        auto executable = workspace.graph;
        if (hierarchyDirty) {
            if (executable) cuda_check(cudaGraphExecDestroy(executable));
            workspace.graph = nullptr;
            cudaGraph_t graph;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int iteration = 0; iteration < 8; ++iteration) cuda::iterate_pressure(workspace, parameters, stream);
            cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&executable, graph, 0));
            cuda_check(cudaGraphDestroy(graph));
            workspace.graph = executable;
            hierarchyDirty  = false;
        }
        cuda::IterationState status{};
        for (;;) {
            int remaining = config.pressureMaxIterations - status.iterations;
            if (remaining >= 8) cuda_check(cudaGraphLaunch(executable, stream));
            else
                for (int i = 0; i < remaining; ++i) cuda::iterate_pressure(workspace, parameters, stream);
            cuda_check(cudaMemcpyAsync(&status, workspace.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaStreamSynchronize(stream));
            if (status.failed) break;
            if (status.converged || status.iterations >= config.pressureMaxIterations) {
                status.converged = 0;
                cuda_check(cudaMemcpyAsync(workspace.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream));
                cuda::check_pressure(workspace, parameters, stream);
                cuda_check(cudaMemcpyAsync(&status, workspace.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream));
                cuda_check(cudaStreamSynchronize(stream));
                if (status.maxResidual <= config.divergenceTolerance * 0.5f || status.failed || status.iterations >= config.pressureMaxIterations) break;
                // Restart from the true residual when FP32 recursive CG loses accuracy.
                status.converged = 0;
                cuda_check(cudaMemcpyAsync(workspace.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream));
                cuda::begin_pressure(workspace, parameters, stream);
            }
        }
        if (status.failed || status.maxResidual > config.divergenceTolerance * 0.5f) throw std::runtime_error("Pressure solve failed: iterations=" + std::to_string(status.iterations) + ", true residual=" + std::to_string(status.maxResidual));
        cuda::apply_pressure(grid, fields, workspace.solution, config.cellSize, parameters.wind, stream);
        cuda_check(cudaStreamSynchronize(stream));
        statistics.projectionMilliseconds += std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        statistics.pressureIterations += status.iterations;
        statistics.pressureLevels = workspace.levelCount;
    }

} // namespace pyro::sparse
