#include "solver.cuh"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <cuda/algorithm>
#include <cuda/std/span>

#include "advection.cuh"
#include "flow_map.cuh"

namespace pyro::lfm {
    Solver::Solver(const Config& settings)
        : device(0),
          pool(device),
          stream(device),
          scene(stream, pool.as_ref()),
          config(settings),
          pressure(settings.resolution, settings.pressure, stream, pool.as_ref()),
          host_pool(),
          mask_storage(stream, pool.as_ref()),
          history_storage(stream, pool.as_ref()),
          output_storage(stream, pool.as_ref()),
          collider_storage(stream, pool.as_ref()),
          host_output(stream, host_pool.as_ref()) {
        grid.cells = make_layout(config.resolution);
        history.resize(config.cycle_steps);
        std::size_t mask_count = grid.cells.count;
        for (int axis = 0; axis < 3; ++axis) {
            int3 size = config.resolution;
            if (axis == 0) ++size.x;
            if (axis == 1) ++size.y;
            if (axis == 2) ++size.z;
            grid.faces[axis] = make_layout(size);
            auto count       = static_cast<std::size_t>(grid.faces[axis].count);
            mask_count += count;
            auto& faces  = face_storage.emplace_back(::cuda::make_buffer<float>(stream, pool.as_ref(), count * (config.cycle_steps + 4), ::cuda::no_init));
            auto* cursor = faces.data();
            for (auto* field : {&grid.initial, &grid.work, &grid.impulse, &grid.error}) {
                field->axis[axis] = cursor;
                cursor += count;
            }
            for (auto& field : history) {
                field.axis[axis] = cursor;
                cursor += count;
            }
            auto& maps                   = map_storage.emplace_back(::cuda::make_buffer<float3>(stream, pool.as_ref(), count * 4, ::cuda::no_init));
            grid.backward.position[axis] = maps.data();
            grid.backward.tangent[axis]  = maps.data() + count;
            grid.forward.position[axis]  = maps.data() + count * 2;
            grid.forward.tangent[axis]   = maps.data() + count * 3;
        }
        mask_storage = ::cuda::make_buffer<unsigned char>(stream, pool.as_ref(), mask_count, ::cuda::no_init);
        grid.fluid   = mask_storage.data();
        auto* mask   = mask_storage.data() + grid.cells.count;
        for (int axis = 0; axis < 3; ++axis) {
            grid.boundary[axis] = mask;
            mask += grid.faces[axis].count;
        }
        history_storage = ::cuda::make_buffer<Velocity>(stream, pool.as_ref(), history.size(), ::cuda::no_init);
        grid.history    = history_storage.data();
        ::cuda::copy_bytes(stream, ::cuda::std::span(history.data(), history.size()), history_storage, {.src_access_order = ::cuda::source_access_order::stream});
        auto output_count = std::size_t(config.resolution.x) * config.resolution.y * config.resolution.z;
        output_storage    = ::cuda::make_buffer<float>(stream, pool.as_ref(), output_count, ::cuda::no_init);
        grid.output       = output_storage.data();
        initialize(grid, config.wind, stream);
        stream.sync();
    }

    Solver::~Solver() noexcept {
        stream.sync();
    }

    template <bool profiling>
    void Solver::advance(double duration) {
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
        statistics.substeps = config.cycle_steps;
        start();
        prepare_geometry();
        finish(statistics.geometry_milliseconds);
        float dt = static_cast<float>(duration / config.cycle_steps);
        for (int k = 0; k < config.cycle_steps; ++k) {
            auto original  = k == 0 ? grid.initial : k == 1 ? history[0] : history[k - 2];
            auto carrier   = k == 0 ? grid.initial : history[k - 1];
            float interval = k == 0 ? dt * .5f : k == 1 ? dt : 2 * dt;
            start();
            advect(grid, history[k], original, carrier, interval / config.cell_size, config.wind, stream);
            finish(statistics.advection_milliseconds);
            start();
            statistics.pressure_iterations += pressure.project(grid, history[k], config.cell_size);
            finish(statistics.projection_milliseconds);
        }
        start();
        reconstruct(grid, config.cycle_steps, config.cell_size, config.wind, dt, stream);
        finish(statistics.flow_map_milliseconds);
        start();
        statistics.pressure_iterations += pressure.project(grid, grid.work, config.cell_size);
        finish(statistics.projection_milliseconds);
        std::swap(grid.initial, grid.work);
        start();
        measure(grid, grid.initial, config.cell_size, pressure.view.reductions, stream);
        double measured[3];
        ::cuda::copy_bytes(stream, ::cuda::std::span(pressure.view.reductions + 3 * grid.cells.blocks, 3), ::cuda::std::span(measured, 3), {.src_access_order = ::cuda::source_access_order::stream});
        finish(statistics.statistics_milliseconds);
        synchronize();
        statistics.fluid_cells    = static_cast<int>(measured[0]);
        statistics.max_divergence = static_cast<float>(measured[1]);
        statistics.max_speed      = static_cast<float>(measured[2]);
        if constexpr (profiling) {
            events[0][1].record(stream);
            events[0][1].sync();
            statistics.gpu_milliseconds = static_cast<float>((events[0][1] - events[0][0]).count()) * 1.e-6f;
            for (std::size_t i = 0; i < phases.size(); ++i) *phases[i] += static_cast<float>((events[i + 1][1] - events[i + 1][0]).count()) * 1.e-6f;
        }
        statistics.allocated_bytes   = pool.attribute(::cuda::memory_pool_attributes::used_mem_current);
        statistics.reserved_bytes    = pool.attribute(::cuda::memory_pool_attributes::reserved_mem_current);
        statistics.pressure_levels   = pressure.view.level_count;
        statistics.wall_milliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        time += duration;
    }

    template void Solver::advance<false>(double);

    template void Solver::advance<true>(double);

    void Solver::synchronize() {
        stream.sync();
    }

    void Solver::checkpoint(const char* path, bool restore) {
        prepare_inputs(0);
        prepare_geometry();
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
                ::cuda::copy_bytes(stream, ::cuda::std::span(values.data(), static_cast<std::size_t>(values.size())), ::cuda::std::span(grid.initial.axis[axis], static_cast<std::size_t>(values.size())), {.src_access_order = ::cuda::source_access_order::stream});
                synchronize();
            } else {
                ::cuda::copy_bytes(stream, ::cuda::std::span(grid.initial.axis[axis], static_cast<std::size_t>(values.size())), ::cuda::std::span(values.data(), static_cast<std::size_t>(values.size())), {.src_access_order = ::cuda::source_access_order::stream});
                synchronize();
                file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            }
        }
    }

    std::span<const float> Solver::download_vorticity() {
        prepare_inputs(0);
        prepare_geometry();
        auto count = std::size_t(config.resolution.x) * config.resolution.y * config.resolution.z;
        if (count > host_output.size()) host_output = ::cuda::make_buffer<float>(stream, host_pool.as_ref(), count, ::cuda::no_init);
        export_vorticity(grid, grid.initial, config.cell_size, stream);
        ::cuda::copy_bytes(stream, ::cuda::std::span(grid.output, count), host_output.first(count), {.src_access_order = ::cuda::source_access_order::stream});
        synchronize();
        return {host_output.data(), count};
    }

    void Solver::prepare_inputs(double duration) {
        scene.prepare(0, 0);
        bool colliders_changed = colliders != collider_inputs;
        for (std::size_t i = 0; i < colliders.size(); ++i) {
            const auto& instance = scene.prepared[colliders[i].instance];
            colliders_changed |= i >= collider_instances.size() || instance != collider_instances[i];
        }
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

    void Solver::prepare_geometry() {
        if (!geometry_dirty) return;
        update_geometry(grid, config.cell_size, config.origin, collider_storage.data(), static_cast<int>(colliders.size()), stream);
        enforce_boundary(grid, grid.initial, config.wind, stream);
        pressure.build(grid);
        geometry_dirty = false;
    }

}
