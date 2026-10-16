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
        if (auto cuda_result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        cudaMemPool_t pool;
        if (auto cuda_result = cudaDeviceGetDefaultMemPool(&pool, 0); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        unsigned long long threshold = ~0ull;
        if (auto cuda_result = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        try {
            pressure.levels = new cuda::sparse::PressureLevel[32]{};
            allocate(topology.counters, 3);
            allocate(pressure.iteration, 1);
            reserve(config.initial_blocks);
        } catch (...) {
            release_storage();
            throw;
        }
    }
    Solver::~Solver() {
        release_storage();
    }

    void Solver::set_sources(const Source* data, int count) {
        bool changed = source_inputs.size() != static_cast<std::size_t>(count);
        for (int i = 0; i < count && !changed; ++i) {
            const auto& current   = data[i];
            const auto& previous  = source_inputs[i];
            bool geometry_changed = current.shape != previous.shape;
            if (!geometry_changed) {
                if (current.shape == Shape::mesh) geometry_changed = current.mesh != previous.mesh || current.motion.swept.lower != previous.motion.swept.lower || current.motion.swept.upper != previous.motion.swept.upper || !std::ranges::equal(current.motion.matrix, previous.motion.matrix) || std::ranges::any_of(current.motion.derivative, [](float value) { return value != 0; }) || std::ranges::any_of(previous.motion.derivative, [](float value) { return value != 0; });
                else geometry_changed = current.center != previous.center || current.half_extent != previous.half_extent;
            }
            changed = geometry_changed || current.velocity != previous.velocity || current.smoke_rate != previous.smoke_rate || current.temperature_rate != previous.temperature_rate || current.velocity_rate != previous.velocity_rate || current.start_time != previous.start_time || current.end_time != previous.end_time || current.inherit_velocity != previous.inherit_velocity || current.motion.max_speed != previous.motion.max_speed;
        }
        if (!changed && source_count == count) return;
        topology_dirty |= changed;
        source_inputs.assign_range(std::span(data, static_cast<std::size_t>(count)));
        if (count > source_capacity) {
            release(sources, source_capacity);
            allocate(sources, count);
            source_capacity = count;
        }
        source_count = count;
        if (!count) return;
        std::vector<cuda::Source> upload(count);
        for (int i = 0; i < count; ++i) {
            const auto& source = data[i];
            upload[i]          = {std::bit_cast<float3>(source.center), std::bit_cast<float3>(source.half_extent), std::bit_cast<float3>(source.velocity), source.smoke_rate, source.temperature_rate, source.velocity_rate, source.start_time, source.end_time, static_cast<cuda::Shape>(source.shape), source.shape == Shape::mesh ? meshes[source.mesh] : cuda::Mesh{}, std::bit_cast<cuda::Motion>(source.motion), source.inherit_velocity};
        }
        if (auto cuda_result = cudaMemcpyAsync(sources, upload.data(), count * sizeof(cuda::Source), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        synchronize();
    }
    void Solver::set_colliders(const Collider* data, int count) {
        bool changed    = moving_geometry || collider_inputs.size() != static_cast<std::size_t>(count);
        moving_geometry = false;
        for (int i = 0; i < count; ++i) {
            const auto& current = data[i];
            if (current.shape == Shape::mesh) moving_geometry |= std::ranges::any_of(current.motion.derivative, [](float value) { return value != 0; });
            else moving_geometry |= current.linear_velocity != std::array<float, 3>{} || current.angular_velocity != std::array<float, 3>{};
            if (!changed) {
                const auto& previous = collider_inputs[i];
                changed              = current.shape != previous.shape || current.mesh != previous.mesh || current.surface_offset != previous.surface_offset;
                if (current.shape == Shape::mesh) changed |= !std::ranges::equal(current.motion.matrix, previous.motion.matrix) || current.motion.swept.lower != previous.motion.swept.lower || current.motion.swept.upper != previous.motion.swept.upper || current.motion.max_speed != previous.motion.max_speed;
                else changed |= current.center != previous.center || current.half_extent != previous.half_extent || current.orientation != previous.orientation;
            }
        }
        changed |= moving_geometry;
        grid.static_geometry = spare.static_geometry = !moving_geometry;
        if (!changed && collider_count == count) return;
        topology_dirty |= changed;
        geometry_dirty |= changed;
        double speed = 0;
        for (int i = 0; i < count; ++i) {
            const auto& collider = data[i];
            double bound         = collider.motion.max_speed;
            if (collider.shape != Shape::mesh) {
                double linear = 0, angular = 0, extent = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    linear += double(collider.linear_velocity[axis]) * collider.linear_velocity[axis];
                    angular += double(collider.angular_velocity[axis]) * collider.angular_velocity[axis];
                    extent += double(collider.half_extent[axis]) * collider.half_extent[axis];
                }
                bound = std::sqrt(linear) + std::sqrt(angular) * std::sqrt(extent);
            }
            speed = std::max(speed, bound);
        }
        grid.collider_speed = spare.collider_speed = speed > 0 ? std::nextafter(static_cast<float>(speed), std::numeric_limits<float>::infinity()) : 0;
        collider_inputs.assign_range(std::span(data, static_cast<std::size_t>(count)));
        if (count > collider_capacity) {
            release(colliders, collider_capacity);
            allocate(colliders, count);
            collider_capacity = count;
        }
        collider_count = count;
        if (!count) return;
        std::vector<cuda::Collider> upload(count);
        for (int i = 0; i < count; ++i) {
            const auto& collider = data[i];
            upload[i]            = {std::bit_cast<float3>(collider.center), std::bit_cast<float3>(collider.half_extent), std::bit_cast<float4>(collider.orientation), std::bit_cast<float3>(collider.linear_velocity), std::bit_cast<float3>(collider.angular_velocity), collider.pose_time, static_cast<cuda::Shape>(collider.shape), collider.shape == Shape::mesh ? meshes[collider.mesh] : cuda::Mesh{}, std::bit_cast<cuda::Motion>(collider.motion), collider.surface_offset};
        }
        if (auto cuda_result = cudaMemcpyAsync(colliders, upload.data(), count * sizeof(cuda::Collider), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        synchronize();
    }
    int Solver::add_mesh(std::span<const Triangle> triangles) {
        auto mesh = upload_mesh(triangles, config.cell_size * 4, stream);
        meshes.push_back(mesh);
        statistics.allocated_bytes += triangles.size() * (sizeof(cuda::Triangle) + sizeof(int)) + mesh.leaf_count * 2ull * sizeof(cuda::Bounds);
        return static_cast<int>(meshes.size()) - 1;
    }
    void Solver::advance(double duration) {
        auto begin                 = std::chrono::steady_clock::now();
        auto bytes                 = statistics.allocated_bytes;
        statistics                 = {};
        statistics.allocated_bytes = bytes;
        statistics.capacity        = grid.layout.capacity;
        std::vector<float*> phases;
        auto start = [&] {
            if (phases.size() == events.size()) {
                events.push_back({});
                for (auto& event : events.back())
                    if (auto cuda_result = cudaEventCreate(&event); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            }
            if (auto cuda_result = cudaEventRecord(events[phases.size()][0], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        };
        auto finish = [&](float& phase) {
            if (auto cuda_result = cudaEventRecord(events[phases.size()][1], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            phases.push_back(&phase);
        };
        auto measure = [&](bool full) {
            std::array<double, 6> result{};
            cuda::sparse::measure(grid, parameters, sources, source_count, static_cast<float>(time), full, pressure.reductions, stream);
            int count = full ? 6 : 2;
            if (auto cuda_result = cudaMemcpyAsync(result.data(), pressure.reductions + grid.layout.count * static_cast<unsigned long long>(count), count * sizeof(double), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            synchronize();
            return result;
        };
        double end_time = time + duration;
        while (time < end_time) {
            if (geometry_dirty || moving_geometry) {
                start();
                cuda::sparse::sample_geometry(grid, {}, nullptr, colliders, collider_count, static_cast<float>(time), 0, stream);
                cuda::sparse::geometry(grid, colliders, collider_count, static_cast<float>(time), 0, stream);
                finish(statistics.geometry_milliseconds);
                geometry_dirty = false;
                matrix_dirty   = true;
            }
            if (topology_dirty || substep % config.adapt_interval == 0) {
                start();
                bool changed;
                for (;;) {
                    int count = cuda::sparse::adapt(grid, spare, topology, scratch, parameters, sources, source_count, colliders, collider_count, static_cast<float>(time), config.max_time_step * config.adapt_interval, changed, stream);
                    if (count <= grid.layout.capacity) {
                        spare.layout.count = count;
                        break;
                    }
                    reserve(count);
                }
                if (changed) {
                    cuda::sparse::build_connectivity(spare, stream);
                    cuda::sparse::sample_geometry(spare, grid.layout, grid.static_geometry ? grid.distance[0] : nullptr, colliders, collider_count, static_cast<float>(time), 0, stream);
                    cuda::sparse::geometry(spare, colliders, collider_count, static_cast<float>(time), 0, stream);
                    cuda::sparse::remap(grid, spare, parameters, stream);
                    std::swap(grid, spare);
                    matrix_dirty = true;
                } else if (grid.layout.count)
                    if (auto cuda_result = cudaMemcpyAsync(grid.layout.blocks, spare.layout.blocks, grid.layout.count * sizeof(cuda::sparse::Block), cudaMemcpyDeviceToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                topology_dirty = false;
                finish(statistics.topology_milliseconds);
                if (changed && grid.layout.count) {
                    start();
                    project(grid.state);
                    finish(statistics.projection_milliseconds);
                }
            }
            if (!grid.layout.count) {
                time = source_count ? std::min(end_time, time + config.max_time_step) : end_time;
                ++substep;
                ++statistics.substeps;
                continue;
            }
            start();
            auto bounds = measure(false);
            finish(statistics.statistics_milliseconds);
            float distance         = config.cfl * config.cell_size;
            float safe_step        = 2.f * distance / (static_cast<float>(bounds[0]) + std::sqrt(static_cast<float>(bounds[0] * bounds[0] + 4 * bounds[1] * distance)) + 1.e-20f);
            double step            = std::min(end_time - time, static_cast<double>(std::min(config.max_time_step, safe_step)));
            float dt               = static_cast<float>(step);
            grid.geometry_time     = static_cast<float>(time);
            grid.geometry_duration = dt;
            if (moving_geometry) {
                start();
                for (int sample = 1; sample < 3; ++sample) cuda::sparse::sample_geometry(grid, {}, nullptr, colliders, collider_count, static_cast<float>(time) + dt * (.5f * sample), sample, stream);
                finish(statistics.geometry_milliseconds);
            }
            for (int stage = 0; stage < 2; ++stage) {
                float stage_step = stage ? dt : .5f * dt;
                auto destination = stage ? grid.next : grid.midpoint;
                auto carrier     = stage ? grid.midpoint : grid.state;
                start();
                cuda::sparse::advect(grid, destination, carrier, parameters, collider_count != 0, static_cast<float>(time), stage_step, stream);
                finish(statistics.advection_milliseconds);
                start();
                cuda::sparse::forces(grid, destination, carrier, parameters, sources, source_count, static_cast<float>(time), stage_step, stream);
                finish(statistics.forces_milliseconds);
                if (moving_geometry) {
                    start();
                    cuda::sparse::geometry(grid, colliders, collider_count, static_cast<float>(time) + stage_step, stage + 1, stream);
                    finish(statistics.geometry_milliseconds);
                    matrix_dirty = true;
                }
                start();
                project(destination);
                finish(statistics.projection_milliseconds);
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
        finish(statistics.statistics_milliseconds);
        statistics.blocks               = grid.layout.count;
        statistics.fluid_cells          = static_cast<int>(result[0]);
        statistics.max_divergence       = static_cast<float>(result[1]);
        statistics.max_speed            = static_cast<float>(result[2]);
        statistics.smoke_integral       = result[3];
        statistics.temperature_integral = result[4];
        if (auto cuda_result = cudaEventSynchronize(events[phases.size() - 1][1]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        for (std::size_t i = 0; i < phases.size(); ++i) {
            float ms;
            if (auto cuda_result = cudaEventElapsedTime(&ms, events[i][0], events[i][1]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            *phases[i] += ms;
        }
        statistics.step_milliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (!std::isfinite(statistics.max_divergence) || statistics.max_divergence > config.divergence_tolerance) throw std::runtime_error("Sparse projection divergence " + std::to_string(statistics.max_divergence) + " exceeds " + std::to_string(config.divergence_tolerance));
    }
    void Solver::synchronize() {
        if (auto cuda_result = cudaGetLastError(); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
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
        auto stored_format = format;
        transfer(stored_format.data(), stored_format.size());
        if (stored_format != format) throw std::runtime_error("Unsupported sparse checkpoint format");
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
            if (auto cuda_result = cudaMemcpyAsync(blocks.data(), grid.layout.blocks, blocks.size() * sizeof(GridBlock), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            synchronize();
        }
        transfer(blocks.data(), blocks.size() * sizeof(GridBlock));
        if (restore)
            if (auto cuda_result = cudaMemcpyAsync(grid.layout.blocks, blocks.data(), blocks.size() * sizeof(GridBlock), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        int field = 0;
        for (auto pointer : {grid.state.smoke, grid.state.temperature, grid.state.velocity}) {
            std::vector<float> buffer(grid.layout.count * (field++ < 2 ? 512ull : 1728ull));
            if (!restore) {
                if (auto cuda_result = cudaMemcpyAsync(buffer.data(), pointer, buffer.size() * sizeof(float), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                synchronize();
            }
            transfer(buffer.data(), buffer.size() * sizeof(float));
            if (restore) {
                if (auto cuda_result = cudaMemcpyAsync(pointer, buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                synchronize();
            }
        }
        int saved_sources = static_cast<int>(source_inputs.size());
        transfer(&saved_sources, sizeof(saved_sources));
        if (restore) source_inputs.resize(saved_sources);
        if (saved_sources) transfer(source_inputs.data(), saved_sources * sizeof(Source));
        int saved_colliders = static_cast<int>(collider_inputs.size());
        transfer(&saved_colliders, sizeof(saved_colliders));
        if (restore) collider_inputs.resize(saved_colliders);
        if (saved_colliders) transfer(collider_inputs.data(), saved_colliders * sizeof(Collider));
        if (restore) {
            cuda::sparse::rebuild_hash(grid.layout, stream);
            cuda::sparse::build_connectivity(grid, stream);
            const auto restored_sources   = source_inputs;
            const auto restored_colliders = collider_inputs;
            source_count = collider_count = -1;
            set_sources(restored_sources.data(), saved_sources);
            set_colliders(restored_colliders.data(), saved_colliders);
            geometry_dirty = matrix_dirty = true;
            topology_dirty                = false;
        }
    }
    std::span<const DensityBlock> Solver::download_density() {
        density.resize(grid.layout.count);
        if (density.empty()) return density;
        cuda::sparse::DensityBlock* device{};
        if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&device), density.size() * sizeof(DensityBlock), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        cuda::sparse::pack_density(grid, device, stream);
        if (auto cuda_result = cudaMemcpyAsync(density.data(), device, density.size() * sizeof(DensityBlock), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaFreeAsync(device, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        synchronize();
        return density;
    }
    template <typename T>
    void Solver::allocate(T*& pointer, unsigned long long count, unsigned long long zero_from) {
        if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&pointer), count * sizeof(T), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (zero_from < count)
            if (auto cuda_result = cudaMemsetAsync(pointer + zero_from, 0, (count - zero_from) * sizeof(T), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        statistics.allocated_bytes += count * sizeof(T);
    }
    template <typename T>
    void Solver::release(T*& pointer, unsigned long long count) {
        if (!pointer) return;
        if (auto cuda_result = cudaFreeAsync(pointer, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        pointer = nullptr;
        statistics.allocated_bytes -= count * sizeof(T);
    }
    template <typename T>
    void Solver::resize(T*& pointer, unsigned long long previous_count, unsigned long long count, unsigned long long retained) {
        if (!retained) {
            release(pointer, previous_count);
            allocate(pointer, count);
            return;
        }
        T* previous = pointer;
        pointer     = nullptr;
        try {
            allocate(pointer, count, retained);
            if (auto cuda_result = cudaMemcpyAsync(pointer, previous, retained * sizeof(T), cudaMemcpyDeviceToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        } catch (...) {
            cudaFreeAsync(previous, stream);
            throw;
        }
        release(previous, previous_count);
    }
    void Solver::reserve(int required) {
        int previous = grid.layout.capacity;
        if (required <= previous) return;
        int capacity        = static_cast<int>(std::bit_ceil(static_cast<unsigned>(std::max(required, previous * 2))));
        int table_size      = static_cast<int>(std::bit_ceil(static_cast<unsigned>(capacity * 8)));
        auto previous_cells = previous * 512ull, cells = capacity * 512ull;
        auto previous_faces = previous * 1728ull, faces = capacity * 1728ull;
        synchronize();
        if (pressure.graph) {
            if (auto cuda_result = cudaGraphExecDestroy(pressure.graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            pressure.graph = nullptr;
        }
        matrix_dirty = true;
        for (auto* g : {&grid, &spare}) {
            auto retained = g == &grid ? static_cast<unsigned long long>(grid.layout.count) : 0;
            resize(g->layout.blocks, previous, capacity, retained);
            resize(g->layout.keys, g->layout.table_size, table_size);
            resize(g->layout.indices, g->layout.table_size, table_size);
            resize(g->faces, previous_faces, faces, retained * 1728);
            resize(g->links, previous * 384ull, capacity * 384ull, retained * 384);
            resize(g->fluid, previous_cells, cells, retained * 512);
            resize(g->solid_divergence, previous_cells, cells, retained * 512);
            for (auto& distance : g->distance) resize(distance, previous * 729ull, capacity * 729ull, retained * 729);
            resize(g->state.smoke, previous_cells, cells, retained * 512);
            resize(g->state.temperature, previous_cells, cells, retained * 512);
            resize(g->state.velocity, previous_faces, faces, retained * 1728);
            g->state.samples     = nullptr;
            g->layout.count      = static_cast<int>(retained);
            g->layout.capacity   = capacity;
            g->layout.table_size = table_size;
            g->layout.levels     = config.levels;
            g->layout.h          = config.cell_size;
            cuda::sparse::rebuild_hash(g->layout, stream);
        }
        for (auto* fields : {&grid.midpoint, &grid.next, &grid.transported}) {
            resize(fields->smoke, previous_cells, cells);
            resize(fields->temperature, previous_cells, cells);
            resize(fields->velocity, previous_faces, faces);
            fields->samples = nullptr;
        }
        spare.midpoint    = grid.midpoint;
        spare.next        = grid.next;
        spare.transported = grid.transported;
        for (int i = 0; i < 3; ++i) {
            resize(grid.samples[i], previous * 5ull * 1331, capacity * 5ull * 1331);
            spare.samples[i] = grid.samples[i];
        }
        resize(grid.uniform_samples, previous * 5ull * 1331, capacity * 5ull * 1331);
        spare.uniform_samples = grid.uniform_samples;
        resize(topology.roots, topology.table_size, table_size);
        resize(topology.refine, topology.table_size, table_size);
        resize(topology.decision, previous, capacity);
        resize(topology.quiet, previous, capacity);
        resize(topology.remaining, previous, capacity);
        resize(topology.split, previous, capacity);
        resize(topology.candidates, previous, capacity);
        topology.table_size = table_size;
        topology.capacity   = capacity;
        for (auto& pointer : scratch.keys) release(pointer, scratch.capacity);
        for (auto& pointer : scratch.values) release(pointer, scratch.capacity);
        release(scratch.count, 1);
        if (scratch.sort_storage) {
            if (auto cuda_result = cudaFreeAsync(scratch.sort_storage, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            statistics.allocated_bytes -= scratch.sort_bytes;
        }
        scratch   = {};
        int edges = capacity * (config.levels > 1 ? 4032 : 1728);
        cuda::sparse::initialize_scratch(scratch, std::max(edges, table_size), stream);
        statistics.allocated_bytes += scratch.capacity * 32ull + scratch.sort_bytes + sizeof(int);
        auto& fine = pressure.levels[0];
        resize(fine.nodes, previous_cells, cells);
        resize(fine.parent, previous_cells, cells);
        resize(fine.edges, fine.edge_capacity, edges);
        resize(fine.face_weights, previous_faces, faces);
        resize(fine.boundary, previous_cells, cells);
        resize(fine.diagonal, previous_cells, cells);
        resize(fine.rhs, previous_cells, cells);
        resize(fine.solution, previous_cells, cells);
        resize(fine.temporary, previous_cells, cells);
        fine.capacity      = static_cast<int>(cells);
        fine.edge_capacity = edges;
        resize(pressure.solution, previous_cells, cells);
        resize(pressure.residual, previous_cells, cells);
        resize(pressure.direction, previous_cells, cells);
        resize(pressure.product, previous_cells, cells);
        resize(pressure.component_sums, previous_cells, cells);
        resize(pressure.components, previous_cells, cells);
        resize(pressure.open, previous_cells, cells);
        resize(pressure.anchors, previous_cells, cells);
        resize(pressure.reductions, (previous + 1ull) * 6, (capacity + 1ull) * 6);
        statistics.capacity = capacity;
        synchronize();
    }
    void Solver::build_pressure() {
        if (pressure.graph) {
            if (auto cuda_result = cudaGraphExecDestroy(pressure.graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            pressure.graph = nullptr;
        }
        auto& fine           = pressure.levels[0];
        fine.count           = grid.layout.count * 512;
        fine.edge_count      = grid.layout.count * (config.levels > 1 ? 4032 : 1728);
        pressure.closed      = cuda::sparse::fine_matrix(grid, pressure, collider_count != 0, stream);
        pressure.level_count = 1;
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
                release(level.edges, level.edge_capacity);
                level.capacity      = count;
                level.edge_capacity = count * 12;
                allocate(level.nodes, count);
                allocate(level.parent, count);
                allocate(level.children, count * 8ull);
                allocate(level.adjacency, count * 24ull);
                allocate(level.edges, level.edge_capacity);
                allocate(level.boundary, count);
                allocate(level.diagonal, count);
                allocate(level.rhs, count);
                allocate(level.solution, count);
                allocate(level.temporary, count);
            }
            level.count = count;
            cuda::sparse::coarse_nodes(previous, level, scratch, depth, stream);
            level.edge_count = cuda::sparse::coarse_edges(previous, level, scratch, stream);
            ++pressure.level_count;
        }
        matrix_dirty = false;
    }
    void Solver::project(cuda::sparse::Fields fields) {
        if (matrix_dirty) build_pressure();
        cuda::sparse::assemble_rhs(grid, fields, pressure, stream);
        if (auto cuda_result = cudaMemsetAsync(pressure.iteration, 0, sizeof(cuda::sparse::IterationState), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        cuda::sparse::begin_pressure(grid, pressure, parameters, stream);
        if (!pressure.graph) {
            cudaGraph_t graph;
            if (auto cuda_result = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            for (int i = 0; i < 8; ++i) cuda::sparse::iterate_pressure(grid, pressure, parameters, stream);
            if (auto cuda_result = cudaStreamEndCapture(stream, &graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaGraphInstantiate(&pressure.graph, graph, 0); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaGraphDestroy(graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        }
        cuda::sparse::IterationState status{};
        for (;;) {
            int remaining = config.pressure_max_iterations - status.iterations;
            if (remaining < 8)
                for (int i = 0; i < remaining; ++i) cuda::sparse::iterate_pressure(grid, pressure, parameters, stream);
            else if (auto cuda_result = cudaGraphLaunch(pressure.graph, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMemcpyAsync(&status, pressure.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            synchronize();
            if (status.failed) break;
            if (status.converged || status.iterations >= config.pressure_max_iterations) {
                status.converged = 0;
                if (auto cuda_result = cudaMemcpyAsync(pressure.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                cuda::sparse::check_pressure(grid, pressure, parameters, stream);
                if (auto cuda_result = cudaMemcpyAsync(&status, pressure.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                synchronize();
                if (status.max_residual <= config.divergence_tolerance * .5f || status.failed || status.iterations >= config.pressure_max_iterations) break;
                status.converged = 0;
                if (auto cuda_result = cudaMemcpyAsync(pressure.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                cuda::sparse::begin_pressure(grid, pressure, parameters, stream);
            }
        }
        if (status.failed || status.max_residual > config.divergence_tolerance * .5f) throw std::runtime_error("Sparse pressure solve failed: iterations=" + std::to_string(status.iterations) + ", true divergence residual=" + std::to_string(status.max_residual));
        cuda::sparse::apply_pressure(grid, fields, pressure.solution, stream);
        statistics.pressure_iterations += status.iterations;
        statistics.pressure_levels = pressure.level_count;
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
            free(g->solid_divergence);
            for (auto distance : g->distance) free(distance);
        }
        for (auto field : {grid.state, spare.state, grid.midpoint, grid.next, grid.transported}) {
            free(field.smoke);
            free(field.temperature);
            free(field.velocity);
        }
        for (auto pointer : grid.samples) free(pointer);
        free(grid.uniform_samples);
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
        free(scratch.sort_storage);
        if (pressure.levels)
            for (int l = 0; l < 32; ++l) {
                auto level = pressure.levels[l];
                free(level.nodes);
                free(level.parent);
                free(level.children);
                free(level.edges);
                free(level.face_weights);
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
        free(pressure.component_sums);
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
