module;
#include "cuda.cuh"
module pyro.dense;
import std;
import pyro.geometry;
namespace pyro::dense {
    static_assert(sizeof(Config) == sizeof(cuda::dense::Config));
    static_assert(sizeof(Motion) == sizeof(cuda::Motion));
    static_assert(sizeof(Triangle) == sizeof(cuda::Triangle));

    Solver::Solver(const Config& settings) : config(settings), parameters(std::bit_cast<cuda::dense::Config>(settings)) {
        if (auto cuda_result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        cudaMemPool_t pool;
        if (auto cuda_result = cudaDeviceGetDefaultMemPool(&pool, 0); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        unsigned long long threshold = ~0ull;
        if (auto cuda_result = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
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
                layout.cells     = (layout.slice * (size.z + 2) + cuda::dense::tile_cells - 1) / cuda::dense::tile_cells * cuda::dense::tile_cells;
                layout.blocks    = layout.cells / cuda::dense::tile_cells;
                level.coarsening = {size.x > 1 ? 2 : 1, size.y > 1 ? 2 : 1, size.z > 1 ? 2 : 1};
                ++pressure.level_count;
                allocate(level.diagonal, layout.cells);
                allocate(level.boundary, layout.cells);
                for (auto& a : level.face) allocate(a, layout.cells);
                allocate(level.solution, layout.cells, l != 0);
                allocate(level.rhs, layout.cells);
                allocate(level.temporary, layout.cells);
                if (size.x * size.y * size.z <= 512) break;
                size = {(size.x + 1) / 2, (size.y + 1) / 2, (size.z + 1) / 2};
            }
            grid.layout = pressure.levels[0].layout;
            auto cells  = grid.layout.cells;
            allocate(grid.fluid, cells);
            allocate(grid.solid_divergence, cells);
            for (auto& phi : grid.distance) allocate(phi, cells);
            for (auto& a : grid.aperture) allocate(a, cells);
            for (auto& u : grid.solid_velocity) allocate(u, cells);
            cuda::dense::Fields* fields[]{&grid.state, &grid.midpoint, &grid.next};
            for (auto field : fields) {
                for (auto& u : field->velocity) allocate(u, cells, false);
                allocate(field->smoke, cells, false);
                allocate(field->temperature, cells, false);
            }
            allocate(pressure.direction, cells, false);
            allocate(pressure.product, cells, false);
            allocate(pressure.solution, cells, false);
            allocate(pressure.residual, cells, false);
            grid.transported.velocity[0] = pressure.direction;
            grid.transported.velocity[1] = pressure.product;
            grid.transported.velocity[2] = pressure.solution;
            grid.transported.smoke       = pressure.residual;
            grid.transported.temperature = pressure.levels[0].solution;

            allocate(pressure.parents, cells);
            allocate(pressure.open_components, cells);
            allocate(pressure.anchors, cells);
            allocate(pressure.reductions, grid.layout.blocks * 6ull + 6);
            allocate(pressure.iteration, 1);
            allocate(maximum, 2);
            for (auto& event : events)
                if (auto cuda_result = cudaEventCreate(&event); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
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
        if (!changed) return;
        if (count > source_capacity) {
            release(sources, source_capacity);
            allocate(sources, count);
            source_capacity = count;
        }
        source_count = count;
        for (int i = count; i < static_cast<int>(source_regions.size()); ++i) release(source_regions[i].weights, source_regions[i].capacity * 4ull);
        source_regions.resize(count);
        if (!count) {
            source_inputs.clear();
            return;
        }
        std::vector<cuda::Source> upload(count);
        std::vector<int> cache;
        for (int i = 0; i < count; ++i) {
            const auto& source = data[i];
            upload[i]          = {std::bit_cast<float3>(source.center), std::bit_cast<float3>(source.half_extent), std::bit_cast<float3>(source.velocity), source.smoke_rate, source.temperature_rate, source.velocity_rate, source.start_time, source.end_time, static_cast<cuda::Shape>(source.shape), source.shape == Shape::mesh ? meshes[source.mesh] : cuda::Mesh{}, std::bit_cast<cuda::Motion>(source.motion), source.inherit_velocity};
            if (i < static_cast<int>(source_inputs.size())) {
                const auto& previous  = source_inputs[i];
                bool geometry_changed = source.shape != previous.shape;
                if (!geometry_changed) {
                    if (source.shape == Shape::mesh) geometry_changed = source.mesh != previous.mesh || source.motion.swept.lower != previous.motion.swept.lower || source.motion.swept.upper != previous.motion.swept.upper || !std::ranges::equal(source.motion.matrix, previous.motion.matrix) || std::ranges::any_of(source.motion.derivative, [](float value) { return value != 0; }) || std::ranges::any_of(previous.motion.derivative, [](float value) { return value != 0; });
                    else geometry_changed = source.center != previous.center || source.half_extent != previous.half_extent;
                }
                if (!geometry_changed) continue;
            }
            auto& region  = source_regions[i];
            region.cached = source.shape != Shape::mesh || std::ranges::all_of(source.motion.derivative, [](float d) { return d == 0; });
            cuda::Bounds bounds;
            if (source.shape == Shape::mesh) bounds = std::bit_cast<cuda::Bounds>(source.motion.swept);
            else {
                auto extent = source.half_extent;
                if (source.shape == Shape::sphere) extent.fill(source.half_extent[0]);
                bounds.lower = {source.center[0] - extent[0], source.center[1] - extent[1], source.center[2] - extent[2]};
                bounds.upper = {source.center[0] + extent[0], source.center[1] + extent[1], source.center[2] + extent[2]};
            }
            auto low = std::bit_cast<std::array<float, 3>>(bounds.lower), high = std::bit_cast<std::array<float, 3>>(bounds.upper);
            std::array<int, 3> lower, extent;
            for (int axis = 0; axis < 3; ++axis) {
                lower[axis]  = std::max(0, int(std::floor((low[axis] - config.origin[axis]) / config.cell_size)) - 1);
                int upper    = std::min(config.resolution[axis], int(std::ceil((high[axis] - config.origin[axis]) / config.cell_size)) + 1);
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
            if (region.cached) cache.push_back(i);
        }
        if (auto cuda_result = cudaMemcpyAsync(sources, upload.data(), count * sizeof(cuda::Source), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        for (int i : cache) cuda::dense::cache_source(grid, parameters, sources, i, source_regions[i], stream);
        source_inputs.assign_range(std::span(data, static_cast<std::size_t>(count)));

        synchronize();
    }

    void Solver::set_colliders(const Collider* data, int count) {
        bool changed         = !grid.static_geometry || collider_inputs.size() != static_cast<std::size_t>(count);
        grid.static_geometry = true;
        for (int i = 0; i < count; ++i) {
            const auto& current = data[i];
            if (current.shape == Shape::mesh) {
                for (float derivative : current.motion.derivative) grid.static_geometry &= derivative == 0.0f;
            } else {
                for (float velocity : current.linear_velocity) grid.static_geometry &= velocity == 0.0f;
                for (float velocity : current.angular_velocity) grid.static_geometry &= velocity == 0.0f;
            }
            if (!changed) {
                const auto& previous = collider_inputs[i];
                changed              = current.shape != previous.shape || current.mesh != previous.mesh || current.surface_offset != previous.surface_offset;
                if (current.shape == Shape::mesh) changed |= !std::ranges::equal(current.motion.matrix, previous.motion.matrix) || current.motion.swept.lower != previous.motion.swept.lower || current.motion.swept.upper != previous.motion.swept.upper || current.motion.max_speed != previous.motion.max_speed;
                else changed |= current.center != previous.center || current.half_extent != previous.half_extent || current.orientation != previous.orientation;
            }
        }
        changed |= !grid.static_geometry;
        if (!changed) return;
        geometry_dirty = true;
        double speed   = 0;
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
        grid.collider_speed = speed > 0 ? std::nextafter(static_cast<float>(speed), std::numeric_limits<float>::infinity()) : 0;
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
        auto elapsed               = [&](int a, int b) {
            float milliseconds;
            if (auto cuda_result = cudaEventElapsedTime(&milliseconds, events[a], events[b]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            return milliseconds;
        };
        double end_time = time + duration;
        while (time < end_time) {
            if (auto cuda_result = cudaEventRecord(events[0], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (!grid.static_geometry || geometry_dirty) {
                std::array<float, 3> low{1.e30f, 1.e30f, 1.e30f}, high{-1.e30f, -1.e30f, -1.e30f};
                for (const auto& body : collider_inputs) {
                    for (int axis = 0; axis < 3; ++axis) {
                        float a, b;
                        if (body.shape == Shape::mesh) {
                            a = body.motion.swept.lower[axis] - std::abs(body.surface_offset);
                            b = body.motion.swept.upper[axis] + std::abs(body.surface_offset);
                        } else {
                            float radius = body.shape == Shape::sphere ? body.half_extent[0] : std::sqrt(body.half_extent[0] * body.half_extent[0] + body.half_extent[1] * body.half_extent[1] + body.half_extent[2] * body.half_extent[2]);
                            float center = body.center[axis] + body.linear_velocity[axis] * (static_cast<float>(time) - body.pose_time);
                            float next   = center + body.linear_velocity[axis] * config.max_time_step;
                            a            = std::min(center, next) - radius;
                            b            = std::max(center, next) + radius;
                        }
                        low[axis]  = std::min(low[axis], a - 4 * config.cell_size);
                        high[axis] = std::max(high[axis], b + 4 * config.cell_size);
                    }
                }
                std::array<int, 3> lower{}, extent{};
                if (collider_count)
                    for (int axis = 0; axis < 3; ++axis) {
                        lower[axis]  = std::max(-1, int(std::floor((low[axis] - config.origin[axis]) / config.cell_size)) - 1);
                        int upper    = std::min(config.resolution[axis], int(std::ceil((high[axis] - config.origin[axis]) / config.cell_size)) + 1);
                        extent[axis] = std::max(0, upper - lower[axis] + 1);
                    }
                grid.geometry_lower  = std::bit_cast<int3>(lower);
                grid.geometry_extent = std::bit_cast<int3>(extent);

                cuda::dense::sample_geometry(grid, config.cell_size, colliders, collider_count, static_cast<float>(time), 0, stream);
                cuda::dense::update_boundary(grid, config.cell_size, colliders, collider_count, static_cast<float>(time), 0, stream);
                geometry_dirty = false;
                matrix_dirty   = true;
            }
            if (auto cuda_result = cudaEventRecord(events[1], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaEventRecord(events[7], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            cuda::dense::measure(grid, grid.state, parameters, sources, source_count, static_cast<float>(time), pressure.reductions, maximum, false, stream);
            if (auto cuda_result = cudaEventRecord(events[8], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            float bounds[2];
            if (auto cuda_result = cudaMemcpyAsync(bounds, maximum, sizeof(bounds), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            synchronize();
            statistics.geometry_milliseconds += elapsed(0, 1);
            statistics.statistics_milliseconds += elapsed(7, 8);
            float distance         = config.cfl * config.cell_size;
            float safe_step        = 2.0f * distance / (bounds[0] + std::sqrt(bounds[0] * bounds[0] + 4.0f * bounds[1] * distance) + 1.0e-20f);
            float dt               = static_cast<float>(std::min(end_time - time, static_cast<double>(std::min(config.max_time_step, safe_step))));
            grid.geometry_time     = static_cast<float>(time);
            grid.geometry_duration = dt;
            if (auto cuda_result = cudaEventRecord(events[0], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (!grid.static_geometry)
                for (int sample = 1; sample < 3; ++sample) cuda::dense::sample_geometry(grid, config.cell_size, colliders, collider_count, static_cast<float>(time) + dt * (0.5f * sample), sample, stream);
            if (auto cuda_result = cudaEventRecord(events[1], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            for (int stage = 0; stage < 2; ++stage) {
                float stage_step = stage ? dt : dt * 0.5f;
                auto destination = stage ? grid.next : grid.midpoint;
                auto carrier     = stage ? grid.midpoint : grid.state;
                if (auto cuda_result = cudaEventRecord(events[2], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                cuda::dense::advect(grid, parameters, destination, grid.state, carrier, collider_count != 0, static_cast<float>(time), stage_step, stream);
                if (auto cuda_result = cudaEventRecord(events[3], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                cuda::dense::apply_sources_forces(grid, parameters, destination, carrier, sources, source_regions.data(), source_count, static_cast<float>(time), stage_step, stream);
                if (auto cuda_result = cudaEventRecord(events[4], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                if (!grid.static_geometry) {
                    cuda::dense::update_boundary(grid, config.cell_size, colliders, collider_count, static_cast<float>(time) + stage_step, stage + 1, stream);
                    matrix_dirty = true;
                }
                if (auto cuda_result = cudaEventRecord(events[5], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                project(destination);
                if (auto cuda_result = cudaEventRecord(events[6], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                if (auto cuda_result = cudaEventSynchronize(events[6]); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                statistics.advection_milliseconds += elapsed(2, 3);
                statistics.forces_milliseconds += elapsed(3, 4);
                statistics.geometry_milliseconds += elapsed(4, 5);
                statistics.projection_milliseconds += elapsed(5, 6);
            }
            statistics.geometry_milliseconds += elapsed(0, 1);
            std::swap(grid.state, grid.next);
            time += dt;
            ++statistics.substeps;
        }
        if (auto cuda_result = cudaEventRecord(events[7], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        cuda::dense::measure(grid, grid.state, parameters, sources, source_count, static_cast<float>(time), pressure.reductions, maximum, true, stream);
        if (auto cuda_result = cudaEventRecord(events[8], stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        double result[6];
        if (auto cuda_result = cudaMemcpyAsync(result, pressure.reductions + grid.layout.blocks * 6ull, sizeof(result), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        synchronize();
        statistics.statistics_milliseconds += elapsed(7, 8);
        statistics.fluid_cells          = static_cast<int>(result[0]);
        statistics.max_divergence       = static_cast<float>(result[1]);
        statistics.max_speed            = static_cast<float>(result[2]);
        statistics.smoke_integral       = result[3];
        statistics.temperature_integral = result[4];
        statistics.step_milliseconds    = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (!std::isfinite(statistics.max_divergence) || statistics.max_divergence > config.divergence_tolerance) throw std::runtime_error("Dense projection divergence " + std::to_string(statistics.max_divergence) + " exceeds " + std::to_string(config.divergence_tolerance));
    }

    void Solver::synchronize() {
        if (auto cuda_result = cudaGetLastError(); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
    }

    void Solver::checkpoint(const char* path, bool restore) {
        synchronize();
        std::fstream file(path, std::ios::binary | (restore ? std::ios::in : std::ios::out | std::ios::trunc));
        file.exceptions(std::ios::badbit | std::ios::failbit);
        auto transfer = [&](void* data, std::size_t bytes) {
            if (restore) file.read(static_cast<char*>(data), bytes);
            else file.write(static_cast<const char*>(data), bytes);
        };
        constexpr std::array format{'P', 'Y', 'R', 'O', 'D', 'N', '0', '1'};
        auto stored_format = format;
        transfer(stored_format.data(), stored_format.size());
        if (stored_format != format) throw std::runtime_error("Unsupported dense checkpoint format");
        transfer(&time, sizeof(time));
        std::vector<float> buffer(grid.layout.cells);
        float* fields[]{grid.state.smoke, grid.state.temperature, grid.state.velocity[0], grid.state.velocity[1], grid.state.velocity[2]};
        for (auto field : fields) {
            if (restore) {
                transfer(buffer.data(), buffer.size() * sizeof(float));
                if (auto cuda_result = cudaMemcpyAsync(field, buffer.data(), buffer.size() * sizeof(float), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            } else {
                if (auto cuda_result = cudaMemcpyAsync(buffer.data(), field, buffer.size() * sizeof(float), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                synchronize();
                transfer(buffer.data(), buffer.size() * sizeof(float));
            }
            if (restore) synchronize();
        }
        if (restore) geometry_dirty = matrix_dirty = true;
    }

    std::span<const float> Solver::download_density() {
        density.resize(static_cast<std::size_t>(config.resolution[0]) * config.resolution[1] * config.resolution[2]);
        cudaMemcpy3DParms copy{};
        copy.srcPtr = make_cudaPitchedPtr(grid.state.smoke, grid.layout.pitch * sizeof(float), grid.layout.pitch * sizeof(float), grid.layout.size.y + 2);
        copy.srcPos = make_cudaPos(sizeof(float), 1, 1);
        copy.dstPtr = make_cudaPitchedPtr(density.data(), config.resolution[0] * sizeof(float), config.resolution[0] * sizeof(float), config.resolution[1]);
        copy.extent = make_cudaExtent(config.resolution[0] * sizeof(float), config.resolution[1], config.resolution[2]);
        copy.kind   = cudaMemcpyDeviceToHost;
        if (auto cuda_result = cudaMemcpy3DAsync(&copy, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        synchronize();
        return density;
    }

    template <typename T>
    void Solver::allocate(T*& pointer, unsigned long long count, bool clear) {
        if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&pointer), count * sizeof(T), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (clear)
            if (auto cuda_result = cudaMemsetAsync(pointer, 0, count * sizeof(T), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        statistics.allocated_bytes += count * sizeof(T);
    }

    template <typename T>
    void Solver::release(T*& pointer, unsigned long long count) {
        if (!pointer) return;
        if (auto cuda_result = cudaFreeAsync(pointer, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        pointer = nullptr;
        statistics.allocated_bytes -= count * sizeof(T);
    }

    void Solver::project(cuda::dense::Fields fields) {
        auto& workspace = pressure;
        cuda::dense::assemble_pressure(grid, fields, workspace, config.cell_size, collider_count != 0, matrix_dirty, stream);
        if (matrix_dirty)
            for (int l = 1; l < workspace.level_count; ++l) cuda::dense::coarse_matrix(workspace.levels[l - 1], workspace.levels[l], stream);
        matrix_dirty = false;
        if (auto cuda_result = cudaMemsetAsync(workspace.iteration, 0, sizeof(cuda::dense::IterationState), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        cuda::dense::begin_pressure(workspace, parameters, stream);
        auto executable = workspace.graph;
        if (!executable) {
            cudaGraph_t graph;
            if (auto cuda_result = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            for (int iteration = 0; iteration < 8; ++iteration) cuda::dense::iterate_pressure(workspace, parameters, stream);
            if (auto cuda_result = cudaStreamEndCapture(stream, &graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaGraphInstantiate(&executable, graph, 0); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaGraphDestroy(graph); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            workspace.graph = executable;
        }
        cuda::dense::IterationState status{};
        for (;;) {
            int remaining = config.pressure_max_iterations - status.iterations;
            if (remaining < 8)
                for (int i = 0; i < remaining; ++i) cuda::dense::iterate_pressure(workspace, parameters, stream);
            else if (auto cuda_result = cudaGraphLaunch(executable, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMemcpyAsync(&status, workspace.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (status.failed) break;
            if (status.converged || status.iterations >= config.pressure_max_iterations) {
                status.converged = 0;
                if (auto cuda_result = cudaMemcpyAsync(workspace.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                cuda::dense::check_pressure(workspace, parameters, stream);
                if (auto cuda_result = cudaMemcpyAsync(&status, workspace.iteration, sizeof(status), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                if (status.max_residual <= config.divergence_tolerance * 0.5f || status.failed || status.iterations >= config.pressure_max_iterations) break;
                status.converged = 0;
                if (auto cuda_result = cudaMemcpyAsync(workspace.iteration, &status, sizeof(status), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                cuda::dense::begin_pressure(workspace, parameters, stream);
            }
        }
        if (status.failed || status.max_residual > config.divergence_tolerance * 0.5f) throw std::runtime_error("Pressure solve failed: iterations=" + std::to_string(status.iterations) + ", true residual=" + std::to_string(status.max_residual));
        cuda::dense::apply_pressure(grid, fields, workspace.solution, config.cell_size, parameters.wind, stream);
        statistics.pressure_iterations += status.iterations;
        statistics.pressure_levels = workspace.level_count;
    }

    void Solver::release_storage() {
        auto free = [stream = stream](void* pointer) {
            if (pointer) cudaFreeAsync(pointer, stream);
        };
        free(grid.fluid);
        free(grid.solid_divergence);
        for (auto p : grid.distance) free(p);
        for (auto p : grid.aperture) free(p);
        for (auto p : grid.solid_velocity) free(p);
        cuda::dense::Fields fields[]{grid.state, grid.midpoint, grid.next};
        for (auto field : fields) {
            free(field.smoke);
            free(field.temperature);
            for (auto p : field.velocity) free(p);
        }
        for (int l = 0; l < pressure.level_count; ++l) {
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
        free(pressure.open_components);
        free(pressure.anchors);
        free(pressure.reductions);
        free(pressure.iteration);
        if (pressure.graph) cudaGraphExecDestroy(pressure.graph);
        for (auto region : source_regions) free(region.weights);
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
