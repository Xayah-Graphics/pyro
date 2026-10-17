#include "solver.cuh"

#include <algorithm>
#include <bit>
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
#include "transfer.cuh"

namespace pyro::sparse {
    Solver::Solver(const Config& settings)
        : device(0),
          pool(device),
          stream(device),
          scene(stream, pool.as_ref()),
          config(settings),
          pressure(settings.pressure, stream, pool.as_ref()),
          host_pool(),
          block_storage(stream, pool.as_ref()),
          key_storage(stream, pool.as_ref()),
          index_storage(stream, pool.as_ref()),
          face_storage(stream, pool.as_ref()),
          link_storage(stream, pool.as_ref()),
          mask_storage(stream, pool.as_ref()),
          field_storage(stream, pool.as_ref()),
          sort_storage(stream, pool.as_ref()),
          source_storage(stream, pool.as_ref()),
          collider_storage(stream, pool.as_ref()),
          source_weight_storage(stream, pool.as_ref()),
          host_output(stream, host_pool.as_ref()),
          density_storage(stream, pool.as_ref()) {
        reserve(config.initial_blocks);
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
        int source_count = static_cast<int>(sources.size()), collider_count = static_cast<int>(colliders.size());
        statistics.capacity   = grid.layout.capacity;
        auto read_measurement = [&](bool full) {
            std::array<double, 6> result{};
            measure(grid, config.forces, source_storage.data(), source_count, static_cast<float>(time), full, pressure.view.reductions, stream);
            int count = full ? 6 : 2;
            ::cuda::copy_bytes(stream, ::cuda::std::span(pressure.view.reductions + grid.layout.count * static_cast<unsigned long long>(count), static_cast<std::size_t>(count)), ::cuda::std::span(result.data(), static_cast<std::size_t>(count)), {.src_access_order = ::cuda::source_access_order::stream});
            synchronize();
            return result;
        };
        double end_time = time + duration;
        while (time < end_time) {
            if (geometry_dirty || grid.boundary_mode == BoundaryMode::moving) {
                start();
                sample_geometry(grid, {}, nullptr, collider_storage.data(), collider_count, static_cast<float>(time), 0, stream);
                update_geometry(grid, collider_storage.data(), collider_count, static_cast<float>(time), 0, stream);
                finish(statistics.geometry_milliseconds);
                geometry_dirty        = false;
                pressure.matrix_dirty = true;
            }
            if (topology_dirty || substep % config.grid.adapt_interval == 0) {
                start();
                bool changed   = adapt(static_cast<float>(time));
                topology_dirty = false;
                finish(statistics.topology_milliseconds);
                if (changed && grid.layout.count) {
                    start();
                    statistics.pressure_iterations += pressure.project(grid, grid.state, scratch);
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
            auto bounds = read_measurement(false);
            finish(statistics.statistics_milliseconds);
            float distance         = config.grid.cfl * config.grid.cell_size;
            float safe_step        = 2.f * distance / (static_cast<float>(bounds[0]) + std::sqrt(static_cast<float>(bounds[0] * bounds[0] + 4 * bounds[1] * distance)) + 1.e-20f);
            double step            = std::min(end_time - time, static_cast<double>(std::min(config.max_time_step, safe_step)));
            float dt               = static_cast<float>(step);
            grid.geometry_time     = static_cast<float>(time);
            grid.geometry_duration = dt;
            if (grid.boundary_mode == BoundaryMode::moving) {
                start();
                for (int sample = 1; sample < 3; ++sample) sample_geometry(grid, {}, nullptr, collider_storage.data(), collider_count, static_cast<float>(time) + dt * (.5f * sample), sample, stream);
                finish(statistics.geometry_milliseconds);
            }
            for (int stage = 0; stage < 2; ++stage) {
                float stage_step = stage ? dt : .5f * dt;
                auto destination = stage ? grid.next : grid.midpoint;
                auto carrier     = stage ? grid.midpoint : grid.state;
                start();
                advect(grid, destination, carrier, config.grid.cell_size, config.grid.wind, static_cast<float>(time), stage_step, stream);
                finish(statistics.advection_milliseconds);
                start();
                const float* weights = nullptr;
                if (source_count && std::ranges::all_of(source_instances, [](const auto& source) { return !source.moving; })) {
                    if (source_weights_dirty) {
                        auto count = sources.size() * grid.layout.capacity * 2240ull;
                        if (source_weight_storage.size() < count) source_weight_storage = ::cuda::make_buffer<float>(stream, pool.as_ref(), count, ::cuda::no_init);
                        sample_sources(grid, source_storage.data(), source_count, static_cast<float>(time), source_weight_storage.data(), stream);
                        source_weights_dirty = false;
                    }
                    weights = source_weight_storage.data();
                }
                forces(grid, destination, carrier, config.forces, source_storage.data(), std::span<const Source>(sources), weights, static_cast<float>(time), stage_step, stream);
                finish(statistics.forces_milliseconds);
                if (grid.boundary_mode == BoundaryMode::moving) {
                    start();
                    update_geometry(grid, collider_storage.data(), collider_count, static_cast<float>(time) + stage_step, stage + 1, stream);
                    finish(statistics.geometry_milliseconds);
                    pressure.matrix_dirty = true;
                }
                start();
                statistics.pressure_iterations += pressure.project(grid, destination, scratch);
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
        auto result = read_measurement(true);
        finish(statistics.statistics_milliseconds);
        statistics.capacity             = grid.layout.capacity;
        statistics.blocks               = grid.layout.count;
        statistics.fluid_cells          = static_cast<int>(result[0]);
        statistics.max_divergence       = static_cast<float>(result[1]);
        statistics.max_speed            = static_cast<float>(result[2]);
        statistics.smoke_integral       = result[3];
        statistics.temperature_integral = result[4];
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
        if (!std::isfinite(statistics.max_divergence) || statistics.max_divergence > config.pressure.divergence_tolerance) throw std::runtime_error("Sparse projection divergence " + std::to_string(statistics.max_divergence) + " exceeds " + std::to_string(config.pressure.divergence_tolerance));
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
        auto transfer = [&](void* data, std::size_t size) {
            if (restore) file.read(static_cast<char*>(data), size);
            else file.write(static_cast<const char*>(data), size);
        };
        constexpr std::array format{'P', 'Y', 'R', 'O', 'S', 'P', '0', '2'};
        auto stored_format = format;
        transfer(stored_format.data(), stored_format.size());
        if (stored_format != format) throw std::runtime_error("Unsupported sparse checkpoint format");
        transfer(&time, sizeof(time));
        transfer(&substep, sizeof(substep));
        int saved_instances = static_cast<int>(scene.instances.size());
        transfer(&saved_instances, sizeof(saved_instances));
        if (restore) scene.instances.resize(saved_instances);
        if (saved_instances) transfer(scene.instances.data(), saved_instances * sizeof(::geometry::Instance));
        int count = grid.layout.count;
        transfer(&count, sizeof(count));
        if (restore) {
            grid.layout.count = 0;
            reserve(count);
            grid.layout.count = count;
        }
        std::vector<Block> blocks(grid.layout.count);
        if (!restore) {
            ::cuda::copy_bytes(stream, ::cuda::std::span(grid.layout.blocks, static_cast<std::size_t>(blocks.size())), ::cuda::std::span(blocks.data(), static_cast<std::size_t>(blocks.size())), {.src_access_order = ::cuda::source_access_order::stream});
            synchronize();
        }
        transfer(blocks.data(), blocks.size() * sizeof(Block));
        if (restore) ::cuda::copy_bytes(stream, ::cuda::std::span(blocks.data(), static_cast<std::size_t>(blocks.size())), ::cuda::std::span(grid.layout.blocks, static_cast<std::size_t>(blocks.size())), {.src_access_order = ::cuda::source_access_order::stream});
        int field = 0;
        for (auto pointer : {grid.state.smoke, grid.state.temperature, grid.state.velocity}) {
            std::vector<float> buffer(grid.layout.count * (field++ < 2 ? 512ull : 1728ull));
            if (!restore) {
                ::cuda::copy_bytes(stream, ::cuda::std::span(pointer, static_cast<std::size_t>(buffer.size())), ::cuda::std::span(buffer.data(), static_cast<std::size_t>(buffer.size())), {.src_access_order = ::cuda::source_access_order::stream});
                synchronize();
            }
            transfer(buffer.data(), buffer.size() * sizeof(float));
            if (restore) {
                ::cuda::copy_bytes(stream, ::cuda::std::span(buffer.data(), static_cast<std::size_t>(buffer.size())), ::cuda::std::span(pointer, static_cast<std::size_t>(buffer.size())), {.src_access_order = ::cuda::source_access_order::stream});
                synchronize();
            }
        }
        int saved_sources = static_cast<int>(sources.size());
        transfer(&saved_sources, sizeof(saved_sources));
        if (restore) sources.resize(saved_sources);
        if (saved_sources) transfer(sources.data(), saved_sources * sizeof(Source));
        int saved_colliders = static_cast<int>(colliders.size());
        transfer(&saved_colliders, sizeof(saved_colliders));
        if (restore) colliders.resize(saved_colliders);
        if (saved_colliders) transfer(colliders.data(), saved_colliders * sizeof(Collider));
        if (restore) {
            rebuild_hash(grid.layout, stream);
            build_connectivity(grid, stream);
            source_instances.clear();
            collider_instances.clear();
            prepare_inputs(0);
            geometry_dirty = pressure.matrix_dirty = pressure.topology_dirty = true;
            topology_dirty                         = false;
        }
    }

    std::span<const DensityBlock> Solver::download_density() {
        auto count = static_cast<std::size_t>(grid.layout.count);
        if (!count) return {};
        if (count > host_output.size()) host_output = ::cuda::make_buffer<DensityBlock>(stream, host_pool.as_ref(), count, ::cuda::no_init);
        if (count > density_storage.size()) density_storage = ::cuda::make_buffer<DensityBlock>(stream, pool.as_ref(), count, ::cuda::no_init);
        pack_density(grid, density_storage.data(), stream);
        ::cuda::copy_bytes(stream, density_storage.first(count), host_output.first(count), {.src_access_order = ::cuda::source_access_order::stream});
        synchronize();
        return {host_output.data(), count};
    }

    void Solver::prepare_inputs(double duration) {
        scene.prepare(static_cast<float>(time), static_cast<float>(time + duration) + config.max_time_step * config.grid.adapt_interval);
        bool sources_changed = sources != source_inputs;
        if (source_instances.size() == sources.size()) {
            for (std::size_t i = 0; i < sources.size(); ++i) sources_changed |= scene.prepared[sources[i].instance] != source_instances[i];
        } else sources_changed = true;
        if (sources_changed) {
            int count = static_cast<int>(sources.size());
            if (count > source_storage.size()) source_storage = ::cuda::make_buffer<SourceView>(stream, pool.as_ref(), count, ::cuda::no_init);
            std::vector<SourceView> upload(count);
            for (int i = 0; i < count; ++i) {
                const auto& source = sources[i];
                upload[i]          = {scene.prepared[sources[i].instance], source.velocity, source.smoke_rate, source.temperature_rate, source.velocity_rate, source.start_time, source.end_time, source.inherit_velocity};
            }
            if (count) ::cuda::copy_bytes(stream, ::cuda::std::span(upload.data(), static_cast<std::size_t>(count)), ::cuda::std::span(source_storage.data(), static_cast<std::size_t>(count)), {.src_access_order = ::cuda::source_access_order::during_api_call});
            source_weights_dirty = true;
            source_inputs = sources;
            source_instances.resize(sources.size());
            for (std::size_t i = 0; i < sources.size(); ++i) source_instances[i] = scene.prepared[sources[i].instance];
            topology_dirty = true;
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
        grid.boundary_mode = spare.boundary_mode = colliders.empty() ? BoundaryMode::none : moving ? BoundaryMode::moving : BoundaryMode::stationary;
        grid.collider_speed = spare.collider_speed = speed;
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
            topology_dirty  = true;
        }
    }

    void Solver::reserve(int required) {
        int previous = grid.layout.capacity;
        if (required <= previous) return;
        int capacity   = static_cast<int>(std::bit_ceil(static_cast<unsigned>(std::max(required, previous * 2))));
        int table_size = static_cast<int>(std::bit_ceil(static_cast<unsigned>(capacity * 8)));
        auto cells = capacity * 512ull, faces = capacity * 1728ull;
        auto distances = capacity * 729ull, samples = capacity * 5ull * 1331;
        int edges            = capacity * (config.grid.levels > 1 ? 4032 : 1728);
        int scratch_capacity = std::max(edges, table_size);
        stream.sync();
        const auto old_grid = grid;
        auto blocks         = ::cuda::make_buffer<Block>(stream, pool.as_ref(), capacity * 3ull, ::cuda::no_init);
        auto keys           = ::cuda::make_buffer<unsigned long long>(stream, pool.as_ref(), table_size * 4ull + scratch_capacity * 3ull, ::cuda::no_init);
        auto indices        = ::cuda::make_buffer<int>(stream, pool.as_ref(), table_size * 2ull + capacity * 4ull + 4, ::cuda::no_init);
        auto face_data      = ::cuda::make_buffer<Face>(stream, pool.as_ref(), faces * 2, ::cuda::no_init);
        auto links          = ::cuda::make_buffer<Links>(stream, pool.as_ref(), capacity * 384ull * 2, ::cuda::no_init);
        auto masks          = ::cuda::make_buffer<unsigned char>(stream, pool.as_ref(), cells * 2 + samples, ::cuda::no_init);
        auto fields         = ::cuda::make_buffer<float>(stream, pool.as_ref(), 6 * (cells * 2 + faces) + cells * 2 + distances * 6 + samples * 3 + scratch_capacity * 2ull, ::cuda::no_init);
        ::cuda::fill_bytes(stream, blocks, 0);
        ::cuda::fill_bytes(stream, keys, 0);
        ::cuda::fill_bytes(stream, indices, 0);
        ::cuda::fill_bytes(stream, face_data, 0);
        ::cuda::fill_bytes(stream, links, 0);
        ::cuda::fill_bytes(stream, masks, 0);
        ::cuda::fill_bytes(stream, fields, 0);
        auto* field = fields.data();
        for (int slot = 0; auto* g : {&grid, &spare}) {
            g->layout.blocks     = blocks.data() + capacity * slot;
            g->layout.keys       = keys.data() + table_size * slot;
            g->layout.indices    = indices.data() + table_size * slot;
            g->layout.count      = slot ? 0 : old_grid.layout.count;
            g->layout.capacity   = capacity;
            g->layout.table_size = table_size;
            g->layout.levels     = config.grid.levels;
            g->layout.h          = config.grid.cell_size;
            g->faces             = face_data.data() + faces * slot;
            g->links             = links.data() + capacity * 384ull * slot;
            g->fluid             = masks.data() + cells * slot;
            g->solid_divergence  = field;
            field += cells;
            for (auto& distance : g->distance) {
                distance = field;
                field += distances;
            }
            ++slot;
        }
        for (auto* state : {&grid.state, &spare.state, &grid.midpoint, &grid.next, &grid.transported}) {
            state->smoke = field;
            field += cells;
            state->temperature = field;
            field += cells;
            state->velocity = field;
            field += faces;
            state->samples = nullptr;
        }
        spare.midpoint    = grid.midpoint;
        spare.next        = grid.next;
        spare.transported = grid.transported;
        for (int i = 0; i < 3; ++i) {
            grid.samples[i] = spare.samples[i] = field;
            field += samples;
        }
        grid.upper.smoke = spare.upper.smoke = field;
        field += cells;
        grid.upper.temperature = spare.upper.temperature = field;
        field += cells;
        grid.upper.velocity = spare.upper.velocity = field;
        field += faces;
        grid.uniform_samples = spare.uniform_samples = masks.data() + cells * 2;
        topology.roots                               = keys.data() + table_size * 2ull;
        topology.refine                              = keys.data() + table_size * 3ull;
        auto* index                                  = indices.data() + table_size * 2ull;
        topology.decision                            = index;
        index += capacity;
        topology.quiet = index;
        index += capacity;
        topology.remaining = index;
        index += capacity;
        topology.split = index;
        index += capacity;
        topology.counters = index;
        index += 3;
        topology.candidates = blocks.data() + capacity * 2ull;
        topology.table_size = table_size;
        topology.capacity   = capacity;
        scratch.capacity    = scratch_capacity;
        for (int i = 0; i < 3; ++i) scratch.keys[i] = keys.data() + table_size * 4ull + scratch_capacity * static_cast<unsigned long long>(i);
        for (auto& values : scratch.values) {
            values = field;
            field += scratch_capacity;
        }
        scratch.count = index++;
        pressure.reserve(grid.layout);
        source_weights_dirty = true;
        auto retain = [&](auto* source, auto* destination, std::size_t count) {
            if (count) ::cuda::copy_bytes(stream, ::cuda::std::span(source, count), ::cuda::std::span(destination, count), {.src_access_order = ::cuda::source_access_order::stream});
        };
        auto retained = static_cast<std::size_t>(old_grid.layout.count);
        retain(old_grid.layout.blocks, grid.layout.blocks, retained);
        retain(old_grid.faces, grid.faces, retained * 1728);
        retain(old_grid.links, grid.links, retained * 384);
        retain(old_grid.fluid, grid.fluid, retained * 512);
        retain(old_grid.solid_divergence, grid.solid_divergence, retained * 512);
        for (int i = 0; i < 3; ++i) retain(old_grid.distance[i], grid.distance[i], retained * 729);
        retain(old_grid.state.smoke, grid.state.smoke, retained * 512);
        retain(old_grid.state.temperature, grid.state.temperature, retained * 512);
        retain(old_grid.state.velocity, grid.state.velocity, retained * 1728);
        block_storage        = std::move(blocks);
        key_storage          = std::move(keys);
        index_storage        = std::move(indices);
        face_storage         = std::move(face_data);
        link_storage         = std::move(links);
        mask_storage         = std::move(masks);
        field_storage        = std::move(fields);
        scratch.sort_bytes   = scratch_bytes(scratch, stream);
        sort_storage         = ::cuda::make_buffer<std::byte>(stream, pool.as_ref(), scratch.sort_bytes, ::cuda::no_init);
        scratch.sort_storage = sort_storage.data();
        rebuild_hash(grid.layout, stream);
        rebuild_hash(spare.layout, stream);
        stream.sync();
    }

    bool Solver::adapt(float time) {
        int source_count = static_cast<int>(sources.size()), collider_count = static_cast<int>(colliders.size());
        bool changed;
        for (;;) {
            int count = adapt_topology(grid, spare, topology, scratch, config.grid, source_storage.data(), source_count, collider_storage.data(), collider_count, time, config.max_time_step * config.grid.adapt_interval, changed, stream);
            if (count <= grid.layout.capacity) {
                spare.layout.count = count;
                break;
            }
            reserve(count);
        }
        if (changed) {
            build_connectivity(spare, stream);
            sample_geometry(spare, grid.layout, grid.boundary_mode != BoundaryMode::moving ? grid.distance[0] : nullptr, collider_storage.data(), collider_count, time, 0, stream);
            update_geometry(spare, collider_storage.data(), collider_count, time, 0, stream);
            remap(grid, spare, config.grid.wind, stream);
            std::swap(grid, spare);
            pressure.matrix_dirty = pressure.topology_dirty = true;
            source_weights_dirty = true;
        } else if (grid.layout.count) ::cuda::copy_bytes(stream, ::cuda::std::span(spare.layout.blocks, static_cast<std::size_t>(grid.layout.count)), ::cuda::std::span(grid.layout.blocks, static_cast<std::size_t>(grid.layout.count)), {.src_access_order = ::cuda::source_access_order::stream});
        return changed;
    }

}
