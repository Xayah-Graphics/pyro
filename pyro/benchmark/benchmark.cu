#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>

#include <solver/dense/solver.cuh>
#include <solver/lfm/solver.cuh>
#include <solver/sparse/solver.cuh>

__global__ void seed_velocity(pyro::sparse::Grid grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < grid.layout.count * 1728 && grid.faces[i].weight > 0) grid.state.velocity[i] += .01f * sinf(i * .013f);
}

__global__ void seed_velocity(pyro::dense::Grid grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grid.layout.cells) return;
    int3 p{i % grid.layout.pitch - 1, i / grid.layout.pitch % (grid.layout.size.y + 2) - 1, i / grid.layout.slice - 1};
    if (p.x > 0 && p.x < grid.layout.size.x && p.y >= 0 && p.y < grid.layout.size.y && p.z >= 0 && p.z < grid.layout.size.z && grid.aperture[0][i] > 0) grid.state.velocity[0][i] += .1f * sinf(p.x * 6.2831853f / grid.layout.size.x) * sinf(p.y * 6.2831853f / grid.layout.size.y);
}

__global__ void seed_velocity(pyro::lfm::Grid grid) {
    int i       = blockIdx.x * blockDim.x + threadIdx.x;
    auto layout = grid.faces[0];
    if (i >= layout.count || grid.boundary[0][i]) return;
    int tile = i >> 9;
    int3 p{(tile % layout.tiles.x) * 8 + (i & 7), (tile / layout.tiles.x % layout.tiles.y) * 8 + ((i >> 3) & 7), (tile / (layout.tiles.x * layout.tiles.y)) * 8 + ((i >> 6) & 7)};
    grid.initial.axis[0][i] += .1f * sinf(p.x * 6.2831853f / grid.cells.size.x) * sinf(p.y * 6.2831853f / grid.cells.size.y);
}

int main(int argc, char** argv) try {
    pyro::sparse::Config sparse_config;
    pyro::dense::Config dense_config;
    pyro::lfm::Config lfm_config;
    lfm_config.wind     = {.5f, 0, 0};
    std::string backend = "sparse", mode = "pressure", scene = "static", source = "default", source_fields = "both", checkpoint, resume;
    int samples = 20, warmup = 3, initial_blocks = 1024, levels = 3;
    int3 resolution{128, 128, 128};
    float cell_size = 1.f / 64;
    bool profiling = false, buoyancy = true, decay = true, origin_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view option = argv[i];
        if (option == "--solver") backend = argv[++i];
        else if (option == "--scene") scene = argv[++i];
        else if (option == "--source") source = argv[++i];
        else if (option == "--source-fields") source_fields = argv[++i];
        else if (option == "--no-buoyancy") buoyancy = false;
        else if (option == "--no-decay") decay = false;
        else if (option == "--mode") mode = argv[++i];
        else if (option == "--samples") samples = std::stoi(argv[++i]);
        else if (option == "--warmup") warmup = std::stoi(argv[++i]);
        else if (option == "--cell-size") cell_size = std::stof(argv[++i]);
        else if (option == "--initial-blocks") initial_blocks = std::stoi(argv[++i]);
        else if (option == "--levels") levels = std::stoi(argv[++i]);
        else if (option == "--origin") {
            dense_config.origin.x = lfm_config.origin.x = std::stof(argv[++i]);
            dense_config.origin.y = lfm_config.origin.y = std::stof(argv[++i]);
            dense_config.origin.z = lfm_config.origin.z = std::stof(argv[++i]);
            origin_set                                  = true;
        } else if (option == "--wind") {
            sparse_config.grid.wind.x = dense_config.wind.x = lfm_config.wind.x = std::stof(argv[++i]);
            sparse_config.grid.wind.y = dense_config.wind.y = lfm_config.wind.y = std::stof(argv[++i]);
            sparse_config.grid.wind.z = dense_config.wind.z = lfm_config.wind.z = std::stof(argv[++i]);
        } else if (option == "--cycle-steps") lfm_config.cycle_steps = std::stoi(argv[++i]);
        else if (option == "--adapt-interval") sparse_config.grid.adapt_interval = std::stoi(argv[++i]);
        else if (option == "--relative-error") sparse_config.grid.relative_error = std::stof(argv[++i]);
        else if (option == "--smoke-error") sparse_config.grid.smoke_absolute_error = std::stof(argv[++i]);
        else if (option == "--temperature-error") sparse_config.grid.temperature_absolute_error = std::stof(argv[++i]);
        else if (option == "--velocity-error") sparse_config.grid.velocity_absolute_error = std::stof(argv[++i]);
        else if (option == "--checkpoint") checkpoint = argv[++i];
        else if (option == "--resume") resume = argv[++i];
        else if (option == "--resolution") {
            resolution.x = std::stoi(argv[++i]);
            resolution.y = std::stoi(argv[++i]);
            resolution.z = std::stoi(argv[++i]);
        } else if (option == "--profile") profiling = true;
        else if (option == "--help") {
            std::cout << "pyro_benchmark --solver sparse|dense|lfm --mode pressure|matrix|pipeline\n"
                         "               --warmup N --samples N --cell-size H --resolution NX NY NZ\n"
                         "               --initial-blocks N --levels N --profile\n"
                         "               --scene none|static|moving|mixed|closed --source none|static|moving\n"
                         "               --source-fields scalars|velocity|both --no-buoyancy --no-decay\n"
                         "               --origin X Y Z --wind X Y Z --cycle-steps N\n"
                         "               --adapt-interval N --relative-error E --smoke-error E\n"
                         "               --temperature-error E --velocity-error E\n"
                         "               --resume PATH --checkpoint PATH\n";
            return 0;
        } else throw std::runtime_error("Unknown option: " + std::string(option));
    }
    if (mode != "pressure" && mode != "matrix" && mode != "pipeline") throw std::runtime_error("Unknown benchmark: " + mode);
    if (scene != "none" && scene != "static" && scene != "moving" && scene != "mixed" && scene != "closed") throw std::runtime_error("Unknown scene: " + scene);
    if (source == "default") source = backend == "lfm" ? "none" : "static";
    if (source != "none" && source != "static" && source != "moving") throw std::runtime_error("Unknown source: " + source);
    if (source_fields != "scalars" && source_fields != "velocity" && source_fields != "both") throw std::runtime_error("Unknown source fields: " + source_fields);
    if (backend == "lfm" && (scene == "moving" || scene == "mixed" || source != "none")) throw std::runtime_error("LFM supports stationary colliders and no sources");
    auto run = [&]<bool profiling_enabled>(auto& solver) {
        constexpr bool is_sparse  = std::same_as<std::remove_cvref_t<decltype(solver)>, pyro::sparse::Solver>;
        constexpr bool is_lfm     = std::same_as<std::remove_cvref_t<decltype(solver)>, pyro::lfm::Solver>;
        constexpr double duration = 1.0 / 24;
        auto prepare              = [&](auto& target) {
            float scale  = .5f * std::min({resolution.x, resolution.y, resolution.z}) * cell_size;
            auto add_box = [&](float3 half_extent, float3 translation, bool moving) {
                int shape = static_cast<int>(target.scene.shapes.size());
                target.scene.shapes.push_back({.kind = geometry::ShapeKind::box, .box = {half_extent}});
                geometry::Instance instance{.shape = shape};
                instance.motion.rigid.translation[0] = translation.x;
                instance.motion.rigid.translation[1] = translation.y;
                instance.motion.rigid.translation[2] = translation.z;
                if (moving) {
                    instance.motion.rigid.linear_velocity[1]  = .25f * scale;
                    instance.motion.rigid.angular_velocity[2] = .5f;
                }
                int index = static_cast<int>(target.scene.instances.size());
                target.scene.instances.push_back(instance);
                target.colliders.push_back({.instance = index});
            };
            if (scene == "static" || scene == "moving" || scene == "mixed") add_box({.15f * scale, .2f * scale, .2f * scale}, {}, scene == "moving");
            if (scene == "mixed") {
                int shape = static_cast<int>(target.scene.shapes.size()), index = static_cast<int>(target.scene.instances.size());
                target.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {.15f * scale}});
                geometry::Instance instance{.shape = shape};
                instance.motion.rigid.translation[0]     = .4f * scale;
                instance.motion.rigid.linear_velocity[1] = -.15f * scale;
                target.scene.instances.push_back(instance);
                target.colliders.push_back({.instance = index});
            }
            if (scene == "closed")
                for (int axis = 0; axis < 3; ++axis)
                    for (int sign : {-1, 1}) {
                        float3 extent{.84f * scale, .84f * scale, .84f * scale}, translation{};
                        if (axis == 0) {
                            extent.x      = .04f * scale;
                            translation.x = sign * .8f * scale;
                        }
                        if (axis == 1) {
                            extent.y      = .04f * scale;
                            translation.y = sign * .8f * scale;
                        }
                        if (axis == 2) {
                            extent.z      = .04f * scale;
                            translation.z = sign * .8f * scale;
                        }
                        add_box(extent, translation, false);
                    }
            if constexpr (!is_lfm) {
                int shape = static_cast<int>(target.scene.shapes.size()), index = static_cast<int>(target.scene.instances.size());
                target.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {.2f * scale}});
                geometry::Instance instance{.shape = shape};
                instance.motion.rigid.translation[0] = -.4f * scale;
                if (source == "moving") instance.motion.rigid.linear_velocity[0] = .1f * scale;
                target.scene.instances.push_back(instance);
                target.sources.push_back({.instance = index, .velocity = {.5f * scale, .3f * scale, 0}, .smoke_rate = source_fields == "velocity" ? 0.f : 1.f, .temperature_rate = source_fields == "velocity" ? 0.f : 3.f, .velocity_rate = source_fields == "scalars" ? 0.f : 4.f, .inherit_velocity = source == "moving" ? 1.f : 0.f});
            }
            if (!resume.empty()) target.checkpoint(resume.c_str(), true);
            target.template advance<profiling_enabled>(duration);
            if constexpr (!is_lfm)
                if (source == "none") {
                    target.sources.clear();
                    target.template advance<profiling_enabled>(duration);
                }
            target.synchronize();
        };
        prepare(solver);
        if constexpr (is_sparse) solver.pressure.build(solver.grid, solver.scratch);
        else if constexpr (is_lfm) solver.pressure.build(solver.grid);
        else solver.pressure.build(solver.grid, solver.config.cell_size);
        std::array<float*, 3> velocity{};
        std::array<std::size_t, 3> sizes{};
        int channels;
        if constexpr (is_sparse) {
            channels    = 1;
            velocity[0] = solver.grid.state.velocity;
            sizes[0]    = std::size_t(solver.grid.layout.count) * 1728;
        } else {
            channels = 3;
            for (int axis = 0; axis < channels; ++axis) {
                if constexpr (is_lfm) {
                    velocity[axis] = solver.grid.initial.axis[axis];
                    sizes[axis]    = solver.grid.faces[axis].count;
                } else {
                    velocity[axis] = solver.grid.state.velocity[axis];
                    sizes[axis]    = solver.grid.layout.cells;
                }
            }
        }
        std::vector<::cuda::buffer<float, ::cuda::mr::device_accessible>> original;
        if (mode == "pressure") {
            ::cuda::launch(solver.stream, ::cuda::distribute<256>(sizes[0]), static_cast<void (*)(decltype(solver.grid))>(seed_velocity), solver.grid);
            for (int axis = 0; axis < channels; ++axis) {
                auto& buffer = original.emplace_back(::cuda::make_buffer<float>(solver.stream, solver.pool.as_ref(), sizes[axis], ::cuda::no_init));
                ::cuda::copy_bytes(solver.stream, ::cuda::std::span(velocity[axis], sizes[axis]), buffer);
            }
        }
        ::cuda::timed_event start(solver.device), finish(solver.device);
        std::cout << "solver,mode,scene,source,source_fields,buoyancy,decay,sample,time,profiling,wall_ms,gpu_interval_ms,iterations,pressure_residual,fluid_cells,blocks,pressure_levels,pool_used_bytes,pool_reserved_bytes,stage_pressure_ms\n";
        auto measure_sample = [&](auto& target, int sample) {
            if (mode == "pressure")
                for (int axis = 0; axis < channels; ++axis) ::cuda::copy_bytes(target.stream, original[axis], ::cuda::std::span(velocity[axis], sizes[axis]));
            target.synchronize();
            auto begin = std::chrono::steady_clock::now();
            start.record(target.stream);
            int iterations = 0;
            if (mode == "pipeline") {
                target.template advance<profiling_enabled>(duration);
                iterations = target.statistics.pressure_iterations;
            } else if (mode == "matrix") {
                if constexpr (is_sparse) target.pressure.build(target.grid, target.scratch);
                else if constexpr (is_lfm) target.pressure.build(target.grid);
                else target.pressure.build(target.grid, target.config.cell_size);
            } else {
                if constexpr (is_lfm) iterations = target.pressure.project(target.grid, target.grid.initial, target.config.cell_size);
                else if constexpr (is_sparse) iterations = target.pressure.project(target.grid, target.grid.state, target.scratch);
                else iterations = target.pressure.project(target.grid, target.grid.state, target.config.cell_size, target.config.wind);
            }
            finish.record(target.stream);
            target.synchronize();
            float wall_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
            float gpu_ms  = static_cast<float>((finish - start).count()) * 1.e-6f;
            if (sample < 0) return;
            float residual = 0;
            if (mode != "matrix") {
                std::remove_cvref_t<decltype(*target.pressure.view.iteration)> state{};
                ::cuda::copy_bytes(target.stream, ::cuda::std::span(target.pressure.view.iteration, 1), ::cuda::std::span(&state, 1), {.src_access_order = ::cuda::source_access_order::stream});
                target.synchronize();
                residual = state.max_residual;
                if constexpr (is_lfm) residual /= target.config.cell_size;
            }
            int blocks = 0;
            if constexpr (is_sparse) blocks = target.grid.layout.count;
            auto used     = target.pool.attribute(::cuda::memory_pool_attributes::used_mem_current);
            auto reserved = target.pool.attribute(::cuda::memory_pool_attributes::reserved_mem_current);
            std::cout << backend << ',' << mode << ',' << scene << ',' << source << ',' << source_fields << ',' << buoyancy << ',' << decay << ',' << sample << ',' << target.time << ',' << profiling_enabled << ',' << wall_ms << ',' << gpu_ms << ',' << iterations << ',' << residual << ',' << target.statistics.fluid_cells << ',' << blocks << ',' << target.pressure.view.level_count << ',' << used << ',' << reserved << ',' << (mode == "pipeline" ? target.statistics.projection_milliseconds : 0) << '\n';
            if (sample == samples - 1 && !checkpoint.empty()) target.checkpoint(checkpoint.c_str(), false);
        };
        for (int sample = -warmup; sample < samples; ++sample) {
            if (mode == "pipeline") {
                std::remove_cvref_t<decltype(solver)> target(solver.config);
                prepare(target);
                measure_sample(target, sample);
            } else measure_sample(solver, sample);
        }
    };
    if (backend == "sparse") {
        sparse_config.grid.cell_size = cell_size;
        sparse_config.initial_blocks = initial_blocks;
        sparse_config.grid.levels    = levels;
        if (!buoyancy) sparse_config.forces.buoyancy = sparse_config.forces.smoke_weight = 0;
        if (!decay) sparse_config.forces.cooling = sparse_config.forces.dissipation = 0;
        pyro::sparse::Solver solver(sparse_config);
        if (profiling) run.template operator()<true>(solver);
        else run.template operator()<false>(solver);
    } else if (backend == "dense") {
        dense_config.cell_size = cell_size;
        if (!origin_set) dense_config.origin = {-resolution.x * cell_size * .5f, -resolution.y * cell_size * .5f, -resolution.z * cell_size * .5f};
        dense_config.resolution = resolution;
        if (!buoyancy) dense_config.forces.buoyancy = dense_config.forces.smoke_weight = 0;
        if (!decay) dense_config.forces.cooling = dense_config.forces.dissipation = 0;
        pyro::dense::Solver solver(dense_config);
        if (profiling) run.template operator()<true>(solver);
        else run.template operator()<false>(solver);
    } else if (backend == "lfm") {
        lfm_config.cell_size = cell_size;
        if (!origin_set) lfm_config.origin = {-resolution.x * cell_size * .5f, -resolution.y * cell_size * .5f, -resolution.z * cell_size * .5f};
        lfm_config.resolution = resolution;
        pyro::lfm::Solver solver(lfm_config);
        if (profiling) run.template operator()<true>(solver);
        else run.template operator()<false>(solver);
    } else throw std::runtime_error("Unknown solver: " + backend);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
