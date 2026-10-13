module;
#include "cuda.cuh"
module pyro.dense;
import std;
namespace pyro::dense {
    static_assert(sizeof(Config) == sizeof(cuda::dense::Config));
    static_assert(sizeof(Motion) == sizeof(cuda::Motion));
    static_assert(sizeof(Triangle) == sizeof(cuda::Triangle));
    static void cuda_check(cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    Solver::Solver(const Config& settings) : config(settings), parameters(std::bit_cast<cuda::dense::Config>(settings)) {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaMemPool_t pool;
        cuda_check(cudaDeviceGetDefaultMemPool(&pool, 0));
        unsigned long long threshold = ~0ull;
        cuda_check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
        try {
            grid.origin     = parameters.origin;
            pressure.levels = new cuda::dense::PressureLevel[std::bit_width(static_cast<unsigned>(std::ranges::max(config.resolution)))]{};
            int3 size       = parameters.resolution;
            for (int l = 0; l < std::bit_width(static_cast<unsigned>(std::ranges::max(config.resolution))); ++l) {
                auto& level      = pressure.levels[l];
                auto& layout     = level.layout;
                layout.size      = size;
                layout.pitch     = (size.x + 2 + 31) / 32 * 32;
                layout.slice     = layout.pitch * (size.y + 2);
                layout.cells     = (layout.slice * (size.z + 2) + cuda::dense::tileCells - 1) / cuda::dense::tileCells * cuda::dense::tileCells;
                layout.blocks    = layout.cells / cuda::dense::tileCells;
                level.coarsening = {size.x > 1 ? 2 : 1, size.y > 1 ? 2 : 1, size.z > 1 ? 2 : 1};
                ++pressure.levelCount;
                allocate(level.diagonal, layout.cells);
                allocate(level.boundary, layout.cells);
                for (auto& a : level.face) allocate(a, layout.cells);
                allocate(level.solution, layout.cells);
                allocate(level.rhs, layout.cells);
                allocate(level.temporary, layout.cells);
                if (size.x * size.y * size.z <= 512) break;
                size = {(size.x + 1) / 2, (size.y + 1) / 2, (size.z + 1) / 2};
            }
            grid.layout = pressure.levels[0].layout;
            auto cells  = grid.layout.cells;
            allocate(grid.fluid, cells);
            allocate(grid.solidDivergence, cells);
            for (auto& phi : grid.distance) allocate(phi, cells);
            for (auto& a : grid.aperture) allocate(a, cells);
            for (auto& u : grid.solidVelocity) allocate(u, cells);
            cuda::dense::Fields* fields[]{&grid.state, &grid.midpoint, &grid.next};
            for (auto field : fields) {
                for (auto& u : field->velocity) allocate(u, cells);
                allocate(field->smoke, cells);
                allocate(field->temperature, cells);
            }
            allocate(pressure.direction, cells);
            allocate(pressure.product, cells);
            allocate(pressure.solution, cells);
            allocate(pressure.residual, cells);
            // Advection and projection never overlap. Reuse Krylov work vectors
            // for the forward transport pass; each pressure solve starts afresh.
            grid.transported.velocity[0] = pressure.direction;
            grid.transported.velocity[1] = pressure.product;
            grid.transported.velocity[2] = pressure.solution;
            grid.transported.smoke       = pressure.residual;
            grid.transported.temperature = pressure.levels[0].solution;

            allocate(pressure.parents, cells);
            allocate(pressure.openComponents, cells);
            allocate(pressure.anchors, cells);
            allocate(pressure.reductions, grid.layout.blocks * 6ull + 6);
            allocate(pressure.iteration, 1);
            allocate(maximum, 2);
            for (auto& event : events) cuda_check(cudaEventCreate(&event));
            cuda::dense::initialize_grid(grid, parameters.wind, stream);
            synchronize();
        } catch (...) {
            release_storage();
            throw;
        }
    }

    Solver::~Solver() {
        release_storage();
    }

    void Solver::set_sources(const Source* data, int count) {
        if (count > sourceCapacity) {
            release(sources, sourceCapacity);
            allocate(sources, count);
            sourceCapacity = count;
        }
        sourceCount = count;
        for (int i = count; i < static_cast<int>(sourceRegions.size()); ++i) release(sourceRegions[i].weights, sourceRegions[i].capacity * 4ull);
        sourceRegions.resize(count);
        if (!count) {
            sourceInputs.clear();
            return;
        }
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
            auto& region           = sourceRegions[i];
            region.cached          = source.shape != Shape::mesh || std::ranges::all_of(source.motion.derivative, [](float d) { return d == 0; });
            cuda::Bounds bounds;
            if (source.shape == Shape::mesh) bounds = std::bit_cast<cuda::Bounds>(source.motion.swept);
            else {
                auto extent = source.halfExtent;
                if (source.shape == Shape::sphere) extent.fill(source.halfExtent[0]);
                bounds.lower = {source.center[0] - extent[0], source.center[1] - extent[1], source.center[2] - extent[2]};
                bounds.upper = {source.center[0] + extent[0], source.center[1] + extent[1], source.center[2] + extent[2]};
            }
            auto low = std::bit_cast<std::array<float, 3>>(bounds.lower), high = std::bit_cast<std::array<float, 3>>(bounds.upper);
            std::array<int, 3> lower, extent;
            for (int axis = 0; axis < 3; ++axis) {
                lower[axis]  = std::max(0, int(std::floor((low[axis] - config.origin[axis]) / config.cellSize)) - 1);
                int upper    = std::min(config.resolution[axis], int(std::ceil((high[axis] - config.origin[axis]) / config.cellSize)) + 1);
                extent[axis] = std::max(0, upper - lower[axis] + 1);
            }
            region.lower  = std::bit_cast<int3>(lower);
            region.extent = std::bit_cast<int3>(extent);
            int cells     = extent[0] * extent[1] * extent[2];
            if (region.cached && cells > region.capacity) {
                release(region.weights, region.capacity * 4ull);
                allocate(region.weights, cells * 4ull);
                region.capacity = cells;
            }
        }
        cuda_check(cudaMemcpyAsync(sources, upload.data(), count * sizeof(cuda::Source), cudaMemcpyHostToDevice, stream));
        for (int i = 0; i < count; ++i) {
            const auto& current = data[i];
            bool dirty          = i >= static_cast<int>(sourceInputs.size());
            if (!dirty) {
                const auto& previous = sourceInputs[i];
                dirty                = current.shape != previous.shape || current.mesh != previous.mesh || current.center != previous.center || current.halfExtent != previous.halfExtent || current.motion.swept.lower != previous.motion.swept.lower || current.motion.swept.upper != previous.motion.swept.upper || !std::ranges::equal(current.motion.matrix, previous.motion.matrix) || !std::ranges::equal(current.motion.derivative, previous.motion.derivative);
            }
            if (dirty && sourceRegions[i].cached) cuda::dense::cache_source(grid, parameters, sources, i, sourceRegions[i], stream);
        }
        sourceInputs.assign_range(std::span(data, static_cast<std::size_t>(count)));

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
        auto elapsed              = [&](int a, int b) {
            float milliseconds;
            cuda_check(cudaEventElapsedTime(&milliseconds, events[a], events[b]));
            return milliseconds;
        };
        double endTime = time + duration;
        while (time < endTime) {
            cuda_check(cudaEventRecord(events[0], stream));
            if (!grid.staticGeometry || geometryDirty) {
                std::array<float, 3> low{1.e30f, 1.e30f, 1.e30f}, high{-1.e30f, -1.e30f, -1.e30f};
                for (const auto& body : colliderInputs) {
                    for (int axis = 0; axis < 3; ++axis) {
                        float a, b;
                        if (body.shape == Shape::mesh) {
                            a = body.motion.swept.lower[axis] - std::abs(body.surfaceOffset);
                            b = body.motion.swept.upper[axis] + std::abs(body.surfaceOffset);
                        } else {
                            float radius = body.shape == Shape::sphere ? body.halfExtent[0] : std::sqrt(body.halfExtent[0] * body.halfExtent[0] + body.halfExtent[1] * body.halfExtent[1] + body.halfExtent[2] * body.halfExtent[2]);
                            float center = body.center[axis] + body.linearVelocity[axis] * (static_cast<float>(time) - body.poseTime);
                            float next   = center + body.linearVelocity[axis] * config.maxTimeStep;
                            a            = std::min(center, next) - radius;
                            b            = std::max(center, next) + radius;
                        }
                        low[axis]  = std::min(low[axis], a - 4 * config.cellSize);
                        high[axis] = std::max(high[axis], b + 4 * config.cellSize);
                    }
                }
                std::array<int, 3> lower{}, extent{};
                if (colliderCount)
                    for (int axis = 0; axis < 3; ++axis) {
                        lower[axis]  = std::max(-1, int(std::floor((low[axis] - config.origin[axis]) / config.cellSize)) - 1);
                        int upper    = std::min(config.resolution[axis], int(std::ceil((high[axis] - config.origin[axis]) / config.cellSize)) + 1);
                        extent[axis] = std::max(0, upper - lower[axis] + 1);
                    }
                grid.geometryLower  = std::bit_cast<int3>(lower);
                grid.geometryExtent = std::bit_cast<int3>(extent);

                cuda::dense::sample_geometry(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time), 0, stream);
                cuda::dense::update_boundary(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time), 0, stream);
                geometryDirty = false;
                matrixDirty   = true;
            }
            cuda_check(cudaEventRecord(events[1], stream));
            cuda_check(cudaEventRecord(events[8], stream));
            cuda::dense::measure(grid, grid.state, parameters, sources, sourceCount, colliders, colliderCount, static_cast<float>(time), pressure.reductions, maximum, false, stream);
            cuda_check(cudaEventRecord(events[9], stream));
            float bounds[2];
            cuda_check(cudaMemcpyAsync(bounds, maximum, sizeof(bounds), cudaMemcpyDeviceToHost, stream));
            synchronize();
            statistics.geometryMilliseconds += elapsed(0, 1);
            statistics.statisticsMilliseconds += elapsed(8, 9);
            float distance        = config.cfl * config.cellSize;
            float safeStep        = 2.0f * distance / (bounds[0] + std::sqrt(bounds[0] * bounds[0] + 4.0f * bounds[1] * distance) + 1.0e-20f);
            float dt              = static_cast<float>(std::min(endTime - time, static_cast<double>(std::min(config.maxTimeStep, safeStep))));
            grid.geometryTime     = static_cast<float>(time);
            grid.geometryDuration = dt;
            cuda_check(cudaEventRecord(events[0], stream));
            if (!grid.staticGeometry)
                for (int sample = 1; sample < 3; ++sample) cuda::dense::sample_geometry(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time) + dt * (0.5f * sample), sample, stream);
            cuda_check(cudaEventRecord(events[1], stream));
            for (int stage = 0; stage < 2; ++stage) {
                float stageStep  = stage ? dt : dt * 0.5f;
                auto destination = stage ? grid.next : grid.midpoint;
                auto carrier     = stage ? grid.midpoint : grid.state;
                cuda_check(cudaEventRecord(events[2], stream));
                cuda::dense::advect(grid, parameters, destination, grid.state, carrier, colliders, colliderCount, static_cast<float>(time), stageStep, stream);
                cuda_check(cudaEventRecord(events[3], stream));
                cuda::dense::apply_sources_forces(grid, parameters, destination, carrier, sources, sourceRegions.data(), sourceCount, static_cast<float>(time), stageStep, stream);
                cuda_check(cudaEventRecord(events[4], stream));
                if (!grid.staticGeometry) {
                    cuda::dense::update_boundary(grid, config.cellSize, colliders, colliderCount, static_cast<float>(time) + stageStep, stage + 1, stream);
                    matrixDirty = true;
                }
                cuda_check(cudaEventRecord(events[5], stream));
                project(destination);
                cuda_check(cudaEventRecord(events[6], stream));
                cuda_check(cudaEventSynchronize(events[6]));
                statistics.advectionMilliseconds += elapsed(2, 3);
                statistics.forcesMilliseconds += elapsed(3, 4);
                statistics.geometryMilliseconds += elapsed(4, 5);
                statistics.projectionMilliseconds += elapsed(5, 6);
            }
            statistics.geometryMilliseconds += elapsed(0, 1);
            std::swap(grid.state, grid.next);
            time += dt;
            ++statistics.substeps;
        }
        cuda_check(cudaEventRecord(events[8], stream));
        cuda::dense::measure(grid, grid.state, parameters, sources, sourceCount, colliders, colliderCount, static_cast<float>(time), pressure.reductions, maximum, true, stream);
        cuda_check(cudaEventRecord(events[9], stream));
        double result[6];
        cuda_check(cudaMemcpyAsync(result, pressure.reductions + grid.layout.blocks * 6ull, sizeof(result), cudaMemcpyDeviceToHost, stream));
        synchronize();
        statistics.statisticsMilliseconds += elapsed(8, 9);
        statistics.fluidCells          = static_cast<int>(result[0]);
        statistics.maxDivergence       = static_cast<float>(result[1]);
        statistics.maxSpeed            = static_cast<float>(result[2]);
        statistics.smokeIntegral       = result[3];
        statistics.temperatureIntegral = result[4];
        statistics.stepMilliseconds    = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (!std::isfinite(statistics.maxDivergence) || statistics.maxDivergence > config.divergenceTolerance) throw std::runtime_error("Dense projection divergence " + std::to_string(statistics.maxDivergence) + " exceeds " + std::to_string(config.divergenceTolerance));
    }

    void Solver::synchronize() {
        cuda_check(cudaGetLastError());
        cuda_check(cudaStreamSynchronize(stream));
    }

    void Solver::checkpoint(const char* path, bool restore) {
        synchronize();
        std::fstream file(path, std::ios::binary | (restore ? std::ios::in : std::ios::out | std::ios::trunc));
        file.exceptions(std::ios::badbit | std::ios::failbit);
        auto transfer = [&](void* data, std::size_t bytes) {
            if (restore) file.read(static_cast<char*>(data), bytes);
            else file.write(static_cast<const char*>(data), bytes);
        };
        transfer(&time, sizeof(time));
        std::vector<float> buffer(grid.layout.cells);
        float* fields[]{grid.state.smoke, grid.state.temperature, grid.state.velocity[0], grid.state.velocity[1], grid.state.velocity[2]};
        for (auto field : fields) {
            if (restore) {
                transfer(buffer.data(), buffer.size() * sizeof(float));
                cuda_check(cudaMemcpyAsync(field, buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
            } else {
                cuda_check(cudaMemcpyAsync(buffer.data(), field, buffer.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                synchronize();
                transfer(buffer.data(), buffer.size() * sizeof(float));
            }
            synchronize();
        }
        if (restore) geometryDirty = matrixDirty = true;
    }

    std::span<const float> Solver::download_density() {
        density.resize(static_cast<std::size_t>(config.resolution[0]) * config.resolution[1] * config.resolution[2]);
        cudaMemcpy3DParms copy{};
        copy.srcPtr = make_cudaPitchedPtr(grid.state.smoke, grid.layout.pitch * sizeof(float), grid.layout.pitch * sizeof(float), grid.layout.size.y + 2);
        copy.srcPos = make_cudaPos(sizeof(float), 1, 1);
        copy.dstPtr = make_cudaPitchedPtr(density.data(), config.resolution[0] * sizeof(float), config.resolution[0] * sizeof(float), config.resolution[1]);
        copy.extent = make_cudaExtent(config.resolution[0] * sizeof(float), config.resolution[1], config.resolution[2]);
        copy.kind   = cudaMemcpyDeviceToHost;
        cuda_check(cudaMemcpy3DAsync(&copy, stream));
        synchronize();
        return density;
    }

    template <typename T>
    void Solver::allocate(T*& pointer, unsigned long long count) {
        cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&pointer), count * sizeof(T), stream));
        cuda_check(cudaMemsetAsync(pointer, 0, count * sizeof(T), stream));
        statistics.allocatedBytes += count * sizeof(T);
    }

    template <typename T>
    void Solver::release(T*& pointer, unsigned long long count) {
        if (!pointer) return;
        cuda_check(cudaFreeAsync(pointer, stream));
        pointer = nullptr;
        statistics.allocatedBytes -= count * sizeof(T);
    }

    void Solver::project(cuda::dense::Fields fields) {
        auto& workspace = pressure;
        auto& fine      = workspace.levels[0];
        fine.layout     = grid.layout;
        if (!fine.layout.blocks) return;
        cuda::dense::assemble_pressure(grid, fields, workspace, config.cellSize, colliderCount != 0, matrixDirty, stream);
        if (matrixDirty)
            for (int l = 1; l < workspace.levelCount; ++l) cuda::dense::coarse_matrix(workspace.levels[l - 1], workspace.levels[l], stream);
        matrixDirty = false;
        cuda_check(cudaMemsetAsync(workspace.iteration, 0, sizeof(cuda::dense::IterationState), stream));
        cuda::dense::begin_pressure(workspace, parameters, stream);
        auto executable = workspace.graph;
        if (!executable) {
            if (executable) cuda_check(cudaGraphExecDestroy(executable));
            workspace.graph = nullptr;
            cudaGraph_t graph;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int iteration = 0; iteration < 8; ++iteration) cuda::dense::iterate_pressure(workspace, parameters, stream);
            cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&executable, graph, 0));
            cuda_check(cudaGraphDestroy(graph));
            workspace.graph = executable;
        }
        cuda::dense::IterationState status{};
        for (;;) {
            int remaining = config.pressureMaxIterations - status.iterations;
            if (remaining >= 8) cuda_check(cudaGraphLaunch(executable, stream));
            else
                for (int i = 0; i < remaining; ++i) cuda::dense::iterate_pressure(workspace, parameters, stream);
            cuda_check(cudaMemcpyAsync(&status, workspace.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaStreamSynchronize(stream));
            if (status.failed) break;
            if (status.converged || status.iterations >= config.pressureMaxIterations) {
                status.converged = 0;
                cuda_check(cudaMemcpyAsync(workspace.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream));
                cuda::dense::check_pressure(workspace, parameters, stream);
                cuda_check(cudaMemcpyAsync(&status, workspace.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream));
                cuda_check(cudaStreamSynchronize(stream));
                if (status.maxResidual <= config.divergenceTolerance * 0.5f || status.failed || status.iterations >= config.pressureMaxIterations) break;
                // Restart from the true residual when FP32 recursive CG loses accuracy.
                status.converged = 0;
                cuda_check(cudaMemcpyAsync(workspace.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream));
                cuda::dense::begin_pressure(workspace, parameters, stream);
            }
        }
        if (status.failed || status.maxResidual > config.divergenceTolerance * 0.5f) throw std::runtime_error("Pressure solve failed: iterations=" + std::to_string(status.iterations) + ", true residual=" + std::to_string(status.maxResidual));
        cuda::dense::apply_pressure(grid, fields, workspace.solution, config.cellSize, parameters.wind, stream);
        cuda_check(cudaStreamSynchronize(stream));
        statistics.pressureIterations += status.iterations;
        statistics.pressureLevels = workspace.levelCount;
    }

    void Solver::release_storage() {
        auto free = [stream = stream](void* pointer) {
            if (pointer) cudaFreeAsync(pointer, stream);
        };
        free(grid.fluid);
        free(grid.solidDivergence);
        for (auto p : grid.distance) free(p);
        for (auto p : grid.aperture) free(p);
        for (auto p : grid.solidVelocity) free(p);
        cuda::dense::Fields fields[]{grid.state, grid.midpoint, grid.next};
        for (auto field : fields) {
            free(field.smoke);
            free(field.temperature);
            for (auto p : field.velocity) free(p);
        }
        for (int l = 0; l < pressure.levelCount; ++l) {
            auto& level = pressure.levels[l];
            free(level.diagonal);
            free(level.boundary);
            for (auto p : level.face) free(p);
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
        for (auto region : sourceRegions) free(region.weights);
        free(sources);
        free(colliders);
        for (auto mesh : meshes) {
            free(mesh.triangles);
            free(mesh.nodes);
            free(mesh.order);
        }
        free(maximum);
        cudaStreamSynchronize(stream);
        for (auto event : events)
            if (event) cudaEventDestroy(event);
        cudaStreamDestroy(stream);
    }

} // namespace pyro::dense
