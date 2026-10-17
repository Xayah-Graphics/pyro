#include "solver.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda/algorithm>
#include <cuda/std/span>

#include "advection.cuh"
#include "forces.cuh"

namespace pyro::dense {
    Solver::Solver(const Config& settings)
        : device(0),
          pool(device),
          stream(device),
          scene(stream, pool.as_ref()),
          config(settings),
          pressure(settings.resolution, settings.pressure, stream, pool.as_ref()),
          host_pool(),
          field_storage(stream, pool.as_ref()),
          mask_storage(stream, pool.as_ref()),
          source_storage(stream, pool.as_ref()),
          collider_storage(stream, pool.as_ref()),
          host_output(stream, host_pool.as_ref()) {
        grid.origin   = config.origin;
        grid.layout   = pressure.hierarchy[0].layout;
        auto cells    = static_cast<std::size_t>(grid.layout.cells);
        field_storage = ::cuda::make_buffer<float>(stream, pool.as_ref(), cells * 44 + 2, ::cuda::no_init);
        mask_storage  = ::cuda::make_buffer<unsigned char>(stream, pool.as_ref(), cells * 2, ::cuda::no_init);
        ::cuda::fill_bytes(stream, field_storage.first(cells * 10), 0);
        ::cuda::fill_bytes(stream, mask_storage, 0);
        grid.fluid            = mask_storage.data();
        grid.solid_divergence = field_storage.data();
        for (int axis = 0; axis < 3; ++axis) {
            grid.distance[axis]       = field_storage.data() + cells * (axis + 1);
            grid.aperture[axis]       = field_storage.data() + cells * (axis + 4);
            grid.solid_velocity[axis] = field_storage.data() + cells * (axis + 7);
        }
        auto* cursor = field_storage.data() + cells * 10;
        for (auto* fields : {&grid.state, &grid.midpoint, &grid.next}) {
            for (auto& velocity : fields->velocity) {
                velocity = cursor;
                cursor += cells;
            }
            fields->smoke = cursor;
            cursor += cells;
            fields->temperature = cursor;
            cursor += cells;
        }
        pressure.view.direction = cursor;
        pressure.view.product   = cursor + cells;
        pressure.view.residual  = cursor + cells * 3;
        maximum                 = cursor + cells * 19;
        ::cuda::fill_bytes(stream, ::cuda::std::span(maximum, 2), 0);
        grid.transported      = {{pressure.view.direction, pressure.view.product, cursor + cells * 2}};
        pressure.view.anchors = mask_storage.data() + cells;
        auto* upper = cursor + cells * 4;
        for (auto& velocity : grid.velocity_scratch) {
            velocity = upper;
            upper += cells;
        }
        for (auto& scalar : grid.scalar_scratch) {
            scalar = upper;
            upper += cells * 6;
        }
        initialize_grid(grid, config.wind, stream);
        stream.sync();
    }

    Solver::~Solver() noexcept {
        stream.sync();
    }

    template <bool profiling>
    void Solver::advance(double duration) {
        const std::array<float, 3> origin{config.origin.x, config.origin.y, config.origin.z};
        const std::array<int, 3> resolution{config.resolution.x, config.resolution.y, config.resolution.z};
        auto begin = std::chrono::steady_clock::now();
        statistics = {};
        std::vector<float*> phases;
        if constexpr (profiling) {
            if (events.empty()) events.push_back({::cuda::timed_event(device), ::cuda::timed_event(device)});
            events[0][0].record(stream);
        }
        auto start = [&] {
            if constexpr (profiling) {
                if (phases.size() + 1 == events.size()) events.push_back({::cuda::timed_event(device), ::cuda::timed_event(device)});
                events[phases.size() + 1][0].record(stream);
            }
        };
        auto finish = [&](float& phase) {
            if constexpr (profiling) {
                events[phases.size() + 1][1].record(stream);
                phases.push_back(&phase);
            }
        };
        start();
        prepare_inputs(duration);
        finish(statistics.input_milliseconds);
        int source_count = static_cast<int>(sources.size()), collider_count = static_cast<int>(colliders.size());
        double end_time = time + duration;
        while (time < end_time) {
            start();
            if (grid.boundary_mode == BoundaryMode::moving || geometry_dirty) {
                std::array<float, 3> low{1.e30f, 1.e30f, 1.e30f}, high{-1.e30f, -1.e30f, -1.e30f};
                for (int i = 0; i < collider_count; ++i) {
                    const auto& bounds = collider_instances[i].swept;
                    const std::array a{bounds.lower.x, bounds.lower.y, bounds.lower.z}, b{bounds.upper.x, bounds.upper.y, bounds.upper.z};
                    float padding = std::abs(collider_inputs[i].surface_offset) + 4 * config.cell_size;
                    for (int axis = 0; axis < 3; ++axis) {
                        low[axis]  = std::min(low[axis], a[axis] - padding);
                        high[axis] = std::max(high[axis], b[axis] + padding);
                    }
                }
                std::array<int, 3> lower{}, extent{};
                if (collider_count)
                    for (int axis = 0; axis < 3; ++axis) {
                        lower[axis]  = std::max(-1, int(std::floor((low[axis] - origin[axis]) / config.cell_size)) - 1);
                        int upper    = std::min(resolution[axis], int(std::ceil((high[axis] - origin[axis]) / config.cell_size)) + 1);
                        extent[axis] = std::max(0, upper - lower[axis] + 1);
                    }
                grid.geometry_lower  = int3{lower[0], lower[1], lower[2]};
                grid.geometry_extent = int3{extent[0], extent[1], extent[2]};

                sample_geometry(grid, config.cell_size, collider_storage.data(), collider_count, static_cast<float>(time), 0, stream);
                update_boundary(grid, config.cell_size, collider_storage.data(), collider_count, static_cast<float>(time), 0, stream);
                geometry_dirty        = false;
                pressure.matrix_dirty = true;
            }
            finish(statistics.geometry_milliseconds);
            start();
            measure(grid, grid.state, config.forces, config.cell_size, config.wind, config.max_time_step, source_storage.data(), source_count, static_cast<float>(time), pressure.view.reductions, maximum, false, stream);
            finish(statistics.statistics_milliseconds);
            float bounds[2];
            ::cuda::copy_bytes(stream, ::cuda::std::span(maximum, 2), ::cuda::std::span(bounds, 2), {.src_access_order = ::cuda::source_access_order::stream});
            synchronize();
            float distance         = config.cfl * config.cell_size;
            float safe_step        = 2.0f * distance / (bounds[0] + std::sqrt(bounds[0] * bounds[0] + 4.0f * bounds[1] * distance) + 1.0e-20f);
            float dt               = static_cast<float>(std::min(end_time - time, static_cast<double>(std::min(config.max_time_step, safe_step))));
            grid.geometry_time     = static_cast<float>(time);
            grid.geometry_duration = dt;
            start();
            if (grid.boundary_mode == BoundaryMode::moving)
                for (int sample = 1; sample < 3; ++sample) sample_geometry(grid, config.cell_size, collider_storage.data(), collider_count, static_cast<float>(time) + dt * (0.5f * sample), sample, stream);
            finish(statistics.geometry_milliseconds);
            for (int stage = 0; stage < 2; ++stage) {
                float stage_step = stage ? dt : dt * 0.5f;
                auto destination = stage ? grid.next : grid.midpoint;
                auto carrier     = stage ? grid.midpoint : grid.state;
                start();
                advect(grid, config.cell_size, config.wind, destination, grid.state, carrier, static_cast<float>(time), stage_step, stream);
                finish(statistics.advection_milliseconds);
                start();
                apply_sources_forces(grid, config.forces, config.cell_size, config.wind, destination, carrier, source_regions.data(), std::span<const Source>(sources), std::span<const ::geometry::InstanceView>(source_instances), static_cast<float>(time), stage_step, stream);
                finish(statistics.forces_milliseconds);
                start();
                if (grid.boundary_mode == BoundaryMode::moving) {
                    update_boundary(grid, config.cell_size, collider_storage.data(), collider_count, static_cast<float>(time) + stage_step, stage + 1, stream);
                    pressure.matrix_dirty = true;
                }
                finish(statistics.geometry_milliseconds);
                start();
                statistics.pressure_iterations += pressure.project(grid, destination, config.cell_size, config.wind);
                finish(statistics.projection_milliseconds);
            }
            std::swap(grid.state, grid.next);
            time += dt;
            ++statistics.substeps;
        }
        start();
        measure(grid, grid.state, config.forces, config.cell_size, config.wind, config.max_time_step, source_storage.data(), source_count, static_cast<float>(time), pressure.view.reductions, maximum, true, stream);
        finish(statistics.statistics_milliseconds);
        double result[6];
        ::cuda::copy_bytes(stream, ::cuda::std::span(pressure.view.reductions + grid.layout.blocks * 6ull, 6), ::cuda::std::span(result, 6), {.src_access_order = ::cuda::source_access_order::stream});
        synchronize();
        statistics.fluid_cells          = static_cast<int>(result[0]);
        statistics.max_divergence       = static_cast<float>(result[1]);
        statistics.max_speed            = static_cast<float>(result[2]);
        statistics.smoke_integral       = result[3];
        statistics.temperature_integral = result[4];
        statistics.pressure_levels      = pressure.view.level_count;
        if constexpr (profiling) {
            events[0][1].record(stream);
            events[0][1].sync();
            statistics.gpu_milliseconds = static_cast<float>((events[0][1] - events[0][0]).count()) * 1.e-6f;
            for (std::size_t i = 0; i < phases.size(); ++i) *phases[i] += static_cast<float>((events[i + 1][1] - events[i + 1][0]).count()) * 1.e-6f;
        }
        statistics.allocated_bytes   = pool.attribute(::cuda::memory_pool_attributes::used_mem_current);
        statistics.reserved_bytes    = pool.attribute(::cuda::memory_pool_attributes::reserved_mem_current);
        statistics.wall_milliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (!std::isfinite(statistics.max_divergence) || statistics.max_divergence > config.pressure.divergence_tolerance) throw std::runtime_error("Dense projection divergence " + std::to_string(statistics.max_divergence) + " exceeds " + std::to_string(config.pressure.divergence_tolerance));
    }

    template void Solver::advance<false>(double);

    template void Solver::advance<true>(double);

    void Solver::synchronize() {
        stream.sync();
    }

    void Solver::checkpoint(const char* path, bool restore) {
        synchronize();
        std::fstream file(path, std::ios::binary | (restore ? std::ios::in : std::ios::out | std::ios::trunc));
        file.exceptions(std::ios::badbit | std::ios::failbit);
        auto transfer = [&](void* data, std::size_t bytes) {
            if (restore) file.read(static_cast<char*>(data), bytes);
            else file.write(static_cast<const char*>(data), bytes);
        };
        constexpr std::array format{'P', 'Y', 'R', 'O', 'D', 'N', '0', '2'};
        auto stored_format = format;
        transfer(stored_format.data(), stored_format.size());
        if (stored_format != format) throw std::runtime_error("Unsupported dense checkpoint format");
        transfer(&time, sizeof(time));
        std::vector<float> buffer(grid.layout.cells);
        float* fields[]{grid.state.smoke, grid.state.temperature, grid.state.velocity[0], grid.state.velocity[1], grid.state.velocity[2]};
        for (auto field : fields) {
            if (restore) {
                transfer(buffer.data(), buffer.size() * sizeof(float));
                ::cuda::copy_bytes(stream, ::cuda::std::span(buffer.data(), static_cast<std::size_t>(buffer.size())), ::cuda::std::span(field, static_cast<std::size_t>(buffer.size())), {.src_access_order = ::cuda::source_access_order::stream});
            } else {
                ::cuda::copy_bytes(stream, ::cuda::std::span(field, static_cast<std::size_t>(buffer.size())), ::cuda::std::span(buffer.data(), static_cast<std::size_t>(buffer.size())), {.src_access_order = ::cuda::source_access_order::stream});
                synchronize();
                transfer(buffer.data(), buffer.size() * sizeof(float));
            }
            if (restore) synchronize();
        }
        if (restore) geometry_dirty = pressure.matrix_dirty = true;
    }

    std::span<const float> Solver::download_density() {
        auto count = std::size_t(config.resolution.x) * config.resolution.y * config.resolution.z;
        if (count > host_output.size()) host_output = ::cuda::make_buffer<float>(stream, host_pool.as_ref(), count, ::cuda::no_init);
        cudaMemcpy3DParms copy{};
        copy.srcPtr = make_cudaPitchedPtr(grid.state.smoke, grid.layout.pitch * sizeof(float), grid.layout.pitch * sizeof(float), grid.layout.size.y + 2);
        copy.srcPos = make_cudaPos(sizeof(float), 1, 1);
        copy.dstPtr = make_cudaPitchedPtr(host_output.data(), grid.layout.size.x * sizeof(float), grid.layout.size.x * sizeof(float), grid.layout.size.y);
        copy.extent = make_cudaExtent(grid.layout.size.x * sizeof(float), grid.layout.size.y, grid.layout.size.z);
        copy.kind   = cudaMemcpyDeviceToHost;
        if (auto cuda_result = cudaMemcpy3DAsync(&copy, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        stream.sync();
        return {host_output.data(), count};
    }

    void Solver::prepare_inputs(double duration) {
        scene.prepare(static_cast<float>(time), static_cast<float>(time + duration));
        bool sources_changed = sources != source_inputs;
        if (source_instances.size() == sources.size()) {
            for (std::size_t i = 0; i < sources.size(); ++i) sources_changed |= scene.prepared[sources[i].instance] != source_instances[i];
        } else sources_changed = true;
        if (sources_changed) {
            int count = static_cast<int>(sources.size());
            const std::array origin{config.origin.x, config.origin.y, config.origin.z};
            const std::array resolution{config.resolution.x, config.resolution.y, config.resolution.z};
            if (count > source_storage.size()) source_storage = ::cuda::make_buffer<SourceView>(stream, pool.as_ref(), count, ::cuda::no_init);
            if (count < source_weight_storage.size()) source_weight_storage.erase(source_weight_storage.begin() + count, source_weight_storage.end());
            while (source_weight_storage.size() < count) source_weight_storage.emplace_back(stream, pool.as_ref());
            source_regions.resize(count);
            std::vector<SourceView> upload(count);
            std::vector<int> cache;
            for (int i = 0; i < count; ++i) {
                const auto& source = sources[i];
                upload[i]          = {scene.prepared[sources[i].instance], source.velocity, source.smoke_rate, source.temperature_rate, source.velocity_rate, source.start_time, source.end_time, source.inherit_velocity};
                if (i < static_cast<int>(source_instances.size()) && scene.prepared[sources[i].instance] == source_instances[i]) continue;
                auto& region       = source_regions[i];
                region.cached      = !scene.prepared[sources[i].instance].moving;
                const auto& bounds = scene.prepared[sources[i].instance].swept;
                const std::array low{bounds.lower.x, bounds.lower.y, bounds.lower.z}, high{bounds.upper.x, bounds.upper.y, bounds.upper.z};
                std::array<int, 3> lower, extent;
                for (int axis = 0; axis < 3; ++axis) {
                    lower[axis]  = std::max(0, int(std::floor((low[axis] - origin[axis]) / config.cell_size)) - 1);
                    int upper    = std::min(resolution[axis], int(std::ceil((high[axis] - origin[axis]) / config.cell_size)) + 1);
                    extent[axis] = std::max(0, upper - lower[axis] + 1);
                }
                region.lower  = int3{lower[0], lower[1], lower[2]};
                region.extent = int3{extent[0], extent[1], extent[2]};
                int cells     = extent[0] * extent[1] * extent[2];
                auto& weights = source_weight_storage[i];
                if (region.cached && cells * 4ull > weights.size()) weights = ::cuda::make_buffer<float>(stream, pool.as_ref(), cells * 4ull, ::cuda::no_init);
                region.weights = weights.data();
                if (region.cached) cache.push_back(i);
            }
            if (count) {
                ::cuda::copy_bytes(stream, ::cuda::std::span(upload.data(), static_cast<std::size_t>(count)), ::cuda::std::span(source_storage.data(), static_cast<std::size_t>(count)), {.src_access_order = ::cuda::source_access_order::during_api_call});
                for (int i : cache) cache_source(grid, config.cell_size, upload[i], source_regions[i], stream);
            }
            source_inputs = sources;
            source_instances.resize(sources.size());
            for (std::size_t i = 0; i < sources.size(); ++i) source_instances[i] = scene.prepared[sources[i].instance];
        }
        bool colliders_changed = colliders != collider_inputs;
        bool moving            = false;
        float speed            = 0;
        for (std::size_t i = 0; i < colliders.size(); ++i) {
            const auto& instance = scene.prepared[colliders[i].instance];
            colliders_changed |= i >= collider_instances.size() || instance != collider_instances[i];
            moving |= instance.moving;
            speed = std::max(speed, instance.max_speed);
        }
        grid.boundary_mode  = colliders.empty() ? BoundaryMode::none : moving ? BoundaryMode::moving : BoundaryMode::stationary;
        grid.collider_speed = speed;
        if (colliders_changed) {
            int count = static_cast<int>(colliders.size());
            if (count > collider_storage.size()) collider_storage = ::cuda::make_buffer<ColliderView>(stream, pool.as_ref(), count, ::cuda::no_init);
            std::vector<ColliderView> upload(count);
            collider_instances.resize(count);
            for (int i = 0; i < count; ++i) {
                collider_instances[i] = scene.prepared[colliders[i].instance];
                upload[i]             = {collider_instances[i], colliders[i].surface_offset};
            }
            if (count) ::cuda::copy_bytes(stream, ::cuda::std::span(upload.data(), upload.size()), ::cuda::std::span(collider_storage.data(), upload.size()), {.src_access_order = ::cuda::source_access_order::during_api_call});
            collider_inputs = colliders;
            geometry_dirty  = true;
        }
    }

}
