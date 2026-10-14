module;
#include "cuda.cuh"
module pyro.sparse;
import std;
import pyro.geometry;

namespace pyro::sparse {
    static_assert(sizeof(Config) == sizeof(cuda::sparse::Config));
    static_assert(sizeof(GridBlock) == sizeof(cuda::sparse::Block));
    static_assert(sizeof(DensityBlock) == sizeof(cuda::sparse::DensityBlock));

    Solver::Solver(const Config& settings) : config(settings), parameters(std::bit_cast<cuda::sparse::Config>(settings)) {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaMemPool_t pool;
        cuda_check(cudaDeviceGetDefaultMemPool(&pool, 0));
        unsigned long long threshold = ~0ull;
        cuda_check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
        try {
            pressure.levels = new cuda::sparse::PressureLevel[32]{};
            allocate(topology.counters, 3);
            allocate(pressure.iteration, 1);
            reserve(config.initialBlocks);
        } catch (...) {
            release_storage();
            throw;
        }
    }
    Solver::~Solver() {
        release_storage();
    }

    void Solver::set_sources(const Source* data, int count) {
        bool changed = sources_changed(std::span(data, static_cast<std::size_t>(count)), sourceInputs);
        if (!changed && sourceCount == count) return;
        topologyDirty |= changed;
        sourceInputs.assign_range(std::span(data, static_cast<std::size_t>(count)));
        if (count > sourceCapacity) {
            release(sources, sourceCapacity);
            allocate(sources, count);
            sourceCapacity = count;
        }
        sourceCount = count;
        if (!count) return;
        std::vector<cuda::Source> upload(count);
        for (int i = 0; i < count; ++i) upload[i] = device_input(data[i], meshes);
        cuda_check(cudaMemcpyAsync(sources, upload.data(), count * sizeof(cuda::Source), cudaMemcpyHostToDevice, stream));
        synchronize();
    }
    void Solver::set_colliders(const Collider* data, int count) {
        bool changed   = movingGeometry || colliderInputs.size() != static_cast<std::size_t>(count);
        movingGeometry = false;
        for (int i = 0; i < count; ++i) {
            const auto& current = data[i];
            if (current.shape == Shape::mesh) movingGeometry |= std::ranges::any_of(current.motion.derivative, [](float value) { return value != 0; });
            else movingGeometry |= current.linearVelocity != std::array<float, 3>{} || current.angularVelocity != std::array<float, 3>{};
            if (!changed) {
                const auto& previous = colliderInputs[i];
                changed              = current.shape != previous.shape || current.mesh != previous.mesh || current.surfaceOffset != previous.surfaceOffset;
                if (current.shape == Shape::mesh) changed |= !std::ranges::equal(current.motion.matrix, previous.motion.matrix) || current.motion.swept.lower != previous.motion.swept.lower || current.motion.swept.upper != previous.motion.swept.upper || current.motion.maxSpeed != previous.motion.maxSpeed;
                else changed |= current.center != previous.center || current.halfExtent != previous.halfExtent || current.orientation != previous.orientation;
            }
        }
        changed |= movingGeometry;
        grid.staticGeometry = spare.staticGeometry = !movingGeometry;
        if (!changed && colliderCount == count) return;
        topologyDirty |= changed;
        geometryDirty |= changed;
        grid.colliderSpeed = spare.colliderSpeed = collider_speed(std::span(data, static_cast<std::size_t>(count)));
        colliderInputs.assign_range(std::span(data, static_cast<std::size_t>(count)));
        if (count > colliderCapacity) {
            release(colliders, colliderCapacity);
            allocate(colliders, count);
            colliderCapacity = count;
        }
        colliderCount = count;
        if (!count) return;
        std::vector<cuda::Collider> upload(count);
        for (int i = 0; i < count; ++i) upload[i] = device_input(data[i], meshes);
        cuda_check(cudaMemcpyAsync(colliders, upload.data(), count * sizeof(cuda::Collider), cudaMemcpyHostToDevice, stream));
        synchronize();
    }
    int Solver::add_mesh(std::span<const Triangle> triangles) {
        auto mesh = upload_mesh(triangles, config.cellSize * 4, stream);
        meshes.push_back(mesh);
        statistics.allocatedBytes += triangles.size() * (sizeof(cuda::Triangle) + sizeof(int)) + mesh.leafCount * 2ull * sizeof(cuda::Bounds);
        return static_cast<int>(meshes.size()) - 1;
    }
    void Solver::advance(double duration) {
        auto begin                = std::chrono::steady_clock::now();
        auto bytes                = statistics.allocatedBytes;
        statistics                = {};
        statistics.allocatedBytes = bytes;
        statistics.capacity       = grid.layout.capacity;
        std::vector<float*> phases;
        auto start = [&] {
            if (phases.size() == events.size()) {
                events.push_back({});
                for (auto& event : events.back()) cuda_check(cudaEventCreate(&event));
            }
            cuda_check(cudaEventRecord(events[phases.size()][0], stream));
        };
        auto finish = [&](float& phase) {
            cuda_check(cudaEventRecord(events[phases.size()][1], stream));
            phases.push_back(&phase);
        };
        auto measure = [&](bool full) {
            std::array<double, 6> result{};
            cuda::sparse::measure(grid, parameters, sources, sourceCount, static_cast<float>(time), full, pressure.reductions, stream);
            int count = full ? 6 : 2;
            cuda_check(cudaMemcpyAsync(result.data(), pressure.reductions + grid.layout.count * static_cast<unsigned long long>(count), count * sizeof(double), cudaMemcpyDeviceToHost, stream));
            synchronize();
            return result;
        };
        double endTime = time + duration;
        while (time < endTime) {
            if (geometryDirty || movingGeometry) {
                start();
                cuda::sparse::sample_geometry(grid, {}, nullptr, colliders, colliderCount, static_cast<float>(time), 0, stream);
                cuda::sparse::geometry(grid, colliders, colliderCount, static_cast<float>(time), 0, stream);
                finish(statistics.geometryMilliseconds);
                geometryDirty = false;
                matrixDirty   = true;
            }
            if (topologyDirty || substep % config.adaptInterval == 0) {
                start();
                bool changed;
                for (;;) {
                    int count = cuda::sparse::adapt(grid, spare, topology, scratch, parameters, sources, sourceCount, colliders, colliderCount, static_cast<float>(time), config.maxTimeStep * config.adaptInterval, changed, stream);
                    if (count <= grid.layout.capacity) {
                        spare.layout.count = count;
                        break;
                    }
                    reserve(count);
                }
                if (changed) {
                    cuda::sparse::build_connectivity(spare, stream);
                    cuda::sparse::sample_geometry(spare, grid.layout, grid.staticGeometry ? grid.distance[0] : nullptr, colliders, colliderCount, static_cast<float>(time), 0, stream);
                    cuda::sparse::geometry(spare, colliders, colliderCount, static_cast<float>(time), 0, stream);
                    cuda::sparse::remap(grid, spare, parameters, stream);
                    std::swap(grid, spare);
                    matrixDirty = true;
                } else if (grid.layout.count) cuda_check(cudaMemcpyAsync(grid.layout.blocks, spare.layout.blocks, grid.layout.count * sizeof(cuda::sparse::Block), cudaMemcpyDeviceToDevice, stream));
                topologyDirty = false;
                finish(statistics.topologyMilliseconds);
                if (changed && grid.layout.count) {
                    start();
                    project(grid.state);
                    finish(statistics.projectionMilliseconds);
                }
            }
            if (!grid.layout.count) {
                time = sourceCount ? std::min(endTime, time + config.maxTimeStep) : endTime;
                ++substep;
                ++statistics.substeps;
                continue;
            }
            start();
            auto bounds = measure(false);
            finish(statistics.statisticsMilliseconds);
            float distance        = config.cfl * config.cellSize;
            float safeStep        = 2.f * distance / (static_cast<float>(bounds[0]) + std::sqrt(static_cast<float>(bounds[0] * bounds[0] + 4 * bounds[1] * distance)) + 1.e-20f);
            double step           = std::min(endTime - time, static_cast<double>(std::min(config.maxTimeStep, safeStep)));
            float dt              = static_cast<float>(step);
            grid.geometryTime     = static_cast<float>(time);
            grid.geometryDuration = dt;
            if (movingGeometry) {
                start();
                for (int sample = 1; sample < 3; ++sample) cuda::sparse::sample_geometry(grid, {}, nullptr, colliders, colliderCount, static_cast<float>(time) + dt * (.5f * sample), sample, stream);
                finish(statistics.geometryMilliseconds);
            }
            for (int stage = 0; stage < 2; ++stage) {
                float stageStep  = stage ? dt : .5f * dt;
                auto destination = stage ? grid.next : grid.midpoint;
                auto carrier     = stage ? grid.midpoint : grid.state;
                start();
                cuda::sparse::advect(grid, destination, carrier, parameters, colliderCount != 0, static_cast<float>(time), stageStep, stream);
                finish(statistics.advectionMilliseconds);
                start();
                cuda::sparse::forces(grid, destination, carrier, parameters, sources, sourceCount, static_cast<float>(time), stageStep, stream);
                finish(statistics.forcesMilliseconds);
                if (movingGeometry) {
                    start();
                    cuda::sparse::geometry(grid, colliders, colliderCount, static_cast<float>(time) + stageStep, stage + 1, stream);
                    finish(statistics.geometryMilliseconds);
                    matrixDirty = true;
                }
                start();
                project(destination);
                finish(statistics.projectionMilliseconds);
            }
            std::swap(grid.state, grid.next);
            spare.midpoint    = grid.midpoint;
            spare.next        = grid.next;
            spare.transported = grid.transported;
            time += step;
            ++substep;
            ++statistics.substeps;
        }
        start();
        auto result = measure(true);
        finish(statistics.statisticsMilliseconds);
        statistics.blocks              = grid.layout.count;
        statistics.fluidCells          = static_cast<int>(result[0]);
        statistics.maxDivergence       = static_cast<float>(result[1]);
        statistics.maxSpeed            = static_cast<float>(result[2]);
        statistics.smokeIntegral       = result[3];
        statistics.temperatureIntegral = result[4];
        cuda_check(cudaEventSynchronize(events[phases.size() - 1][1]));
        for (std::size_t i = 0; i < phases.size(); ++i) {
            float ms;
            cuda_check(cudaEventElapsedTime(&ms, events[i][0], events[i][1]));
            *phases[i] += ms;
        }
        statistics.stepMilliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (!std::isfinite(statistics.maxDivergence) || statistics.maxDivergence > config.divergenceTolerance) throw std::runtime_error("Sparse projection divergence " + std::to_string(statistics.maxDivergence) + " exceeds " + std::to_string(config.divergenceTolerance));
    }
    void Solver::synchronize() {
        cuda_check(cudaGetLastError());
        cuda_check(cudaStreamSynchronize(stream));
    }
    void Solver::checkpoint(const char* path, bool restore) {
        synchronize();
        std::fstream file(path, std::ios::binary | (restore ? std::ios::in : std::ios::out | std::ios::trunc));
        file.exceptions(std::ios::badbit | std::ios::failbit);
        auto transfer = [&](void* data, std::size_t size) {
            if (restore) file.read(static_cast<char*>(data), size);
            else file.write(static_cast<const char*>(data), size);
        };
        constexpr std::array format{'P', 'Y', 'R', 'O', 'S', 'P', '0', '1'};
        auto storedFormat = format;
        transfer(storedFormat.data(), storedFormat.size());
        if (storedFormat != format) throw std::runtime_error("Unsupported sparse checkpoint format");
        transfer(&time, sizeof(time));
        transfer(&substep, sizeof(substep));
        int count = grid.layout.count;
        transfer(&count, sizeof(count));
        if (restore) {
            grid.layout.count = 0;
            reserve(count);
            grid.layout.count = count;
        }
        std::vector<GridBlock> blocks(grid.layout.count);
        if (!restore) {
            cuda_check(cudaMemcpyAsync(blocks.data(), grid.layout.blocks, blocks.size() * sizeof(GridBlock), cudaMemcpyDeviceToHost, stream));
            synchronize();
        }
        transfer(blocks.data(), blocks.size() * sizeof(GridBlock));
        if (restore) cuda_check(cudaMemcpyAsync(grid.layout.blocks, blocks.data(), blocks.size() * sizeof(GridBlock), cudaMemcpyHostToDevice, stream));
        int field = 0;
        for (auto pointer : {grid.state.smoke, grid.state.temperature, grid.state.velocity}) {
            std::vector<float> buffer(grid.layout.count * (field++ < 2 ? 512ull : 1728ull));
            if (!restore) {
                cuda_check(cudaMemcpyAsync(buffer.data(), pointer, buffer.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
                synchronize();
            }
            transfer(buffer.data(), buffer.size() * sizeof(float));
            if (restore) {
                cuda_check(cudaMemcpyAsync(pointer, buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                synchronize();
            }
        }
        // Input snapshots preserve adaptation scheduling across a restart while
        // still detecting a changed scene on the first upload after restore.
        int savedSources = static_cast<int>(sourceInputs.size());
        transfer(&savedSources, sizeof(savedSources));
        if (restore) sourceInputs.resize(savedSources);
        if (savedSources) transfer(sourceInputs.data(), savedSources * sizeof(Source));
        int savedColliders = static_cast<int>(colliderInputs.size());
        transfer(&savedColliders, sizeof(savedColliders));
        if (restore) colliderInputs.resize(savedColliders);
        if (savedColliders) transfer(colliderInputs.data(), savedColliders * sizeof(Collider));
        if (restore) {
            cuda::sparse::rebuild_hash(grid.layout, stream);
            cuda::sparse::build_connectivity(grid, stream);
            const auto restoredSources   = sourceInputs;
            const auto restoredColliders = colliderInputs;
            sourceCount = colliderCount = -1;
            set_sources(restoredSources.data(), savedSources);
            set_colliders(restoredColliders.data(), savedColliders);
            geometryDirty = matrixDirty = true;
            topologyDirty               = false;
        }
    }
    std::span<const DensityBlock> Solver::download_density() {
        density.resize(grid.layout.count);
        if (density.empty()) return density;
        cuda::sparse::DensityBlock* device{};
        cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&device), density.size() * sizeof(DensityBlock), stream));
        cuda::sparse::pack_density(grid, device, stream);
        cuda_check(cudaMemcpyAsync(density.data(), device, density.size() * sizeof(DensityBlock), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaFreeAsync(device, stream));
        synchronize();
        return density;
    }
    template <typename T>
    void Solver::allocate(T*& pointer, unsigned long long count, unsigned long long zeroFrom) {
        cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&pointer), count * sizeof(T), stream));
        if (zeroFrom < count) cuda_check(cudaMemsetAsync(pointer + zeroFrom, 0, (count - zeroFrom) * sizeof(T), stream));
        statistics.allocatedBytes += count * sizeof(T);
    }
    template <typename T>
    void Solver::release(T*& pointer, unsigned long long count) {
        if (!pointer) return;
        cuda_check(cudaFreeAsync(pointer, stream));
        pointer = nullptr;
        statistics.allocatedBytes -= count * sizeof(T);
    }
    template <typename T>
    void Solver::resize(T*& pointer, unsigned long long previousCount, unsigned long long count, unsigned long long retained) {
        if (!retained) {
            release(pointer, previousCount);
            allocate(pointer, count);
            return;
        }
        T* previous = pointer;
        pointer     = nullptr;
        try {
            allocate(pointer, count, retained);
            cuda_check(cudaMemcpyAsync(pointer, previous, retained * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        } catch (...) {
            cudaFreeAsync(previous, stream);
            throw;
        }
        release(previous, previousCount);
    }
    void Solver::reserve(int required) {
        int previous = grid.layout.capacity;
        if (required <= previous) return;
        int capacity       = static_cast<int>(std::bit_ceil(static_cast<unsigned>(std::max(required, previous * 2))));
        int tableSize      = static_cast<int>(std::bit_ceil(static_cast<unsigned>(capacity * 8)));
        auto previousCells = previous * 512ull, cells = capacity * 512ull;
        auto previousFaces = previous * 1728ull, faces = capacity * 1728ull;
        synchronize();
        if (pressure.graph) {
            cuda_check(cudaGraphExecDestroy(pressure.graph));
            pressure.graph = nullptr;
        }
        matrixDirty = true;
        // Grow one allocation at a time. Only the live grid is retained; all
        // transport, regrid and pressure workspaces are rebuilt before use.
        for (auto* g : {&grid, &spare}) {
            auto retained = g == &grid ? static_cast<unsigned long long>(grid.layout.count) : 0;
            resize(g->layout.blocks, previous, capacity, retained);
            resize(g->layout.keys, g->layout.tableSize, tableSize);
            resize(g->layout.indices, g->layout.tableSize, tableSize);
            resize(g->faces, previousFaces, faces, retained * 1728);
            resize(g->links, previous * 384ull, capacity * 384ull, retained * 384);
            resize(g->fluid, previousCells, cells, retained * 512);
            resize(g->solidDivergence, previousCells, cells, retained * 512);
            for (auto& distance : g->distance) resize(distance, previous * 729ull, capacity * 729ull, retained * 729);
            resize(g->state.smoke, previousCells, cells, retained * 512);
            resize(g->state.temperature, previousCells, cells, retained * 512);
            resize(g->state.velocity, previousFaces, faces, retained * 1728);
            g->state.samples    = nullptr;
            g->layout.count     = static_cast<int>(retained);
            g->layout.capacity  = capacity;
            g->layout.tableSize = tableSize;
            g->layout.levels    = config.levels;
            g->layout.h         = config.cellSize;
            cuda::sparse::rebuild_hash(g->layout, stream);
        }
        for (auto* fields : {&grid.midpoint, &grid.next, &grid.transported}) {
            resize(fields->smoke, previousCells, cells);
            resize(fields->temperature, previousCells, cells);
            resize(fields->velocity, previousFaces, faces);
            fields->samples = nullptr;
        }
        spare.midpoint    = grid.midpoint;
        spare.next        = grid.next;
        spare.transported = grid.transported;
        for (int i = 0; i < 3; ++i) {
            resize(grid.samples[i], previous * 5ull * 1331, capacity * 5ull * 1331);
            spare.samples[i] = grid.samples[i];
        }
        resize(grid.uniformSamples, previous * 5ull * 1331, capacity * 5ull * 1331);
        spare.uniformSamples = grid.uniformSamples;
        resize(topology.roots, topology.tableSize, tableSize);
        resize(topology.refine, topology.tableSize, tableSize);
        resize(topology.decision, previous, capacity);
        resize(topology.quiet, previous, capacity);
        resize(topology.remaining, previous, capacity);
        resize(topology.split, previous, capacity);
        resize(topology.candidates, previous, capacity);
        topology.tableSize = tableSize;
        topology.capacity  = capacity;
        for (auto& pointer : scratch.keys) release(pointer, scratch.capacity);
        for (auto& pointer : scratch.values) release(pointer, scratch.capacity);
        release(scratch.count, 1);
        if (scratch.sortStorage) {
            cuda_check(cudaFreeAsync(scratch.sortStorage, stream));
            statistics.allocatedBytes -= scratch.sortBytes;
        }
        scratch   = {};
        int edges = capacity * (config.levels > 1 ? 4032 : 1728);
        cuda::sparse::initialize_scratch(scratch, std::max(edges, tableSize), stream);
        statistics.allocatedBytes += scratch.capacity * 32ull + scratch.sortBytes + sizeof(int);
        auto& fine = pressure.levels[0];
        resize(fine.nodes, previousCells, cells);
        resize(fine.parent, previousCells, cells);
        resize(fine.edges, fine.edgeCapacity, edges);
        resize(fine.faceWeights, previousFaces, faces);
        resize(fine.boundary, previousCells, cells);
        resize(fine.diagonal, previousCells, cells);
        resize(fine.rhs, previousCells, cells);
        resize(fine.solution, previousCells, cells);
        resize(fine.temporary, previousCells, cells);
        fine.capacity     = static_cast<int>(cells);
        fine.edgeCapacity = edges;
        resize(pressure.solution, previousCells, cells);
        resize(pressure.residual, previousCells, cells);
        resize(pressure.direction, previousCells, cells);
        resize(pressure.product, previousCells, cells);
        resize(pressure.componentSums, previousCells, cells);
        resize(pressure.components, previousCells, cells);
        resize(pressure.open, previousCells, cells);
        resize(pressure.anchors, previousCells, cells);
        resize(pressure.reductions, (previous + 1ull) * 6, (capacity + 1ull) * 6);
        statistics.capacity = capacity;
        synchronize();
    }
    void Solver::build_pressure() {
        if (pressure.graph) {
            cuda_check(cudaGraphExecDestroy(pressure.graph));
            pressure.graph = nullptr;
        }
        auto& fine          = pressure.levels[0];
        fine.count          = grid.layout.count * 512;
        fine.edgeCount      = grid.layout.count * (config.levels > 1 ? 4032 : 1728);
        pressure.closed     = cuda::sparse::fine_matrix(grid, pressure, colliderCount != 0, stream);
        pressure.levelCount = 1;
        for (int depth = 1; pressure.levels[depth - 1].count > 128; ++depth) {
            auto& previous = pressure.levels[depth - 1];
            auto& level    = pressure.levels[depth];
            int count      = cuda::sparse::coarse_nodes(previous, {}, scratch, depth, stream);
            if (count > level.capacity) {
                release(level.nodes, level.capacity);
                release(level.parent, level.capacity);
                release(level.children, level.capacity * 8ull);
                release(level.adjacency, level.capacity * 24ull);
                release(level.boundary, level.capacity);
                release(level.diagonal, level.capacity);
                release(level.rhs, level.capacity);
                release(level.solution, level.capacity);
                release(level.temporary, level.capacity);
                release(level.edges, level.edgeCapacity);
                level.capacity     = count;
                level.edgeCapacity = count * 12;
                allocate(level.nodes, count);
                allocate(level.parent, count);
                allocate(level.children, count * 8ull);
                allocate(level.adjacency, count * 24ull);
                allocate(level.edges, level.edgeCapacity);
                allocate(level.boundary, count);
                allocate(level.diagonal, count);
                allocate(level.rhs, count);
                allocate(level.solution, count);
                allocate(level.temporary, count);
            }
            level.count = count;
            cuda::sparse::coarse_nodes(previous, level, scratch, depth, stream);
            level.edgeCount = cuda::sparse::coarse_edges(previous, level, scratch, stream);
            ++pressure.levelCount;
        }
        matrixDirty = false;
    }
    void Solver::project(cuda::sparse::Fields fields) {
        if (matrixDirty) build_pressure();
        cuda::sparse::assemble_rhs(grid, fields, pressure, stream);
        cuda_check(cudaMemsetAsync(pressure.iteration, 0, sizeof(cuda::IterationState), stream));
        cuda::sparse::begin_pressure(grid, pressure, parameters, stream);
        if (!pressure.graph) {
            cudaGraph_t graph;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < 8; ++i) cuda::sparse::iterate_pressure(grid, pressure, parameters, stream);
            cuda_check(cudaStreamEndCapture(stream, &graph));
            cuda_check(cudaGraphInstantiate(&pressure.graph, graph, 0));
            cuda_check(cudaGraphDestroy(graph));
        }
        cuda::IterationState status{};
        for (;;) {
            int remaining = config.pressureMaxIterations - status.iterations;
            if (remaining >= 8) cuda_check(cudaGraphLaunch(pressure.graph, stream));
            else
                for (int i = 0; i < remaining; ++i) cuda::sparse::iterate_pressure(grid, pressure, parameters, stream);
            cuda_check(cudaMemcpyAsync(&status, pressure.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream));
            synchronize();
            if (status.failed) break;
            if (status.converged || status.iterations >= config.pressureMaxIterations) {
                status.converged = 0;
                cuda_check(cudaMemcpyAsync(pressure.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream));
                cuda::sparse::check_pressure(grid, pressure, parameters, stream);
                cuda_check(cudaMemcpyAsync(&status, pressure.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream));
                synchronize();
                if (status.maxResidual <= config.divergenceTolerance * .5f || status.failed || status.iterations >= config.pressureMaxIterations) break;
                status.converged = 0;
                cuda_check(cudaMemcpyAsync(pressure.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream));
                cuda::sparse::begin_pressure(grid, pressure, parameters, stream);
            }
        }
        if (status.failed || status.maxResidual > config.divergenceTolerance * .5f) throw std::runtime_error("Sparse pressure solve failed: iterations=" + std::to_string(status.iterations) + ", true divergence residual=" + std::to_string(status.maxResidual));
        cuda::sparse::apply_pressure(grid, fields, pressure.solution, stream);
        statistics.pressureIterations += status.iterations;
        statistics.pressureLevels = pressure.levelCount;
    }
    void Solver::release_storage() {
        auto free = [stream = stream](void* pointer) {
            if (pointer) cudaFreeAsync(pointer, stream);
        };
        if (pressure.graph) cudaGraphExecDestroy(pressure.graph);
        for (auto* g : {&grid, &spare}) {
            free(g->layout.blocks);
            free(g->layout.keys);
            free(g->layout.indices);
            free(g->faces);
            free(g->links);
            free(g->fluid);
            free(g->solidDivergence);
            for (auto distance : g->distance) free(distance);
        }
        for (auto field : {grid.state, spare.state, grid.midpoint, grid.next, grid.transported}) {
            free(field.smoke);
            free(field.temperature);
            free(field.velocity);
        }
        for (auto pointer : grid.samples) free(pointer);
        free(grid.uniformSamples);
        free(topology.roots);
        free(topology.refine);
        free(topology.decision);
        free(topology.quiet);
        free(topology.remaining);
        free(topology.split);
        free(topology.candidates);
        free(topology.counters);
        for (auto pointer : scratch.keys) free(pointer);
        for (auto pointer : scratch.values) free(pointer);
        free(scratch.count);
        free(scratch.sortStorage);
        if (pressure.levels)
            for (int l = 0; l < 32; ++l) {
                auto level = pressure.levels[l];
                free(level.nodes);
                free(level.parent);
                free(level.children);
                free(level.edges);
                free(level.faceWeights);
                free(level.adjacency);
                free(level.boundary);
                free(level.diagonal);
                free(level.rhs);
                free(level.solution);
                free(level.temporary);
            }
        delete[] pressure.levels;
        free(pressure.solution);
        free(pressure.residual);
        free(pressure.direction);
        free(pressure.product);
        free(pressure.componentSums);
        free(pressure.components);
        free(pressure.open);
        free(pressure.anchors);
        free(pressure.reductions);
        free(pressure.iteration);
        free(sources);
        free(colliders);
        for (auto mesh : meshes) {
            free(mesh.triangles);
            free(mesh.order);
            free(mesh.nodes);
        }
        for (auto pair : events)
            for (auto event : pair)
                if (event) cudaEventDestroy(event);
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
} // namespace pyro::sparse
