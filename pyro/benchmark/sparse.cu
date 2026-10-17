#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>
#include <solver/sparse/solver.cuh>

__global__ void seed_velocity(pyro::sparse::Grid grid, float amplitude) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= grid.layout.count * 1728 || grid.faces[f].weight <= 0) return;
    float3 p;
    float h;
    int axis, owner, sign;
    pyro::sparse::face_position(grid.layout, f, p, h, axis, owner, sign);
    float a = axis == 0 ? p.x : axis == 1 ? p.y : p.z;
    float b = axis == 0 ? p.y : axis == 1 ? p.z : p.x;
    float c = axis == 0 ? p.z : axis == 1 ? p.x : p.y;
    grid.state.velocity[f] = amplitude * (sinf(3.141592654f * a) * cosf(1.570796327f * b) * cosf(1.570796327f * c) + .2f * sinf(12.56637061f * a) * sinf(9.424777961f * b) * cosf(6.283185307f * c));
}

int main(int argc, char** argv) try {
    pyro::sparse::Config config;
    config.initial_blocks = 256;
    config.forces.up = {0, 0, 1};
    std::string scene = "static", mode = "pressure", checkpoint_in, checkpoint_out, geometry_path;
    int samples = 12, warmup = 3, age = 0, source_count = 1;
    double duration = 1.0 / 24;
    float amplitude = .03f, source_speed = 0;
    bool profiling = false, audit = false, cold = false;
    for (int i = 1; i < argc; ++i) {
        std::string option = argv[i];
        if (option == "--scene") scene = argv[++i];
        else if (option == "--mode") mode = argv[++i];
        else if (option == "--samples") samples = std::stoi(argv[++i]);
        else if (option == "--warmup") warmup = std::stoi(argv[++i]);
        else if (option == "--sources") source_count = std::stoi(argv[++i]);
        else if (option == "--source-speed") source_speed = std::stof(argv[++i]);
        else if (option == "--age") age = std::stoi(argv[++i]);
        else if (option == "--cell-size") config.grid.cell_size = std::stof(argv[++i]);
        else if (option == "--levels") config.grid.levels = std::stoi(argv[++i]);
        else if (option == "--initial-blocks") config.initial_blocks = std::stoi(argv[++i]);
        else if (option == "--geometry") geometry_path = argv[++i];
        else if (option == "--checkpoint-in") checkpoint_in = argv[++i];
        else if (option == "--checkpoint-out") checkpoint_out = argv[++i];
        else if (option == "--amplitude") amplitude = std::stof(argv[++i]);
        else if (option == "--duration") duration = std::stod(argv[++i]);
        else if (option == "--relative-tolerance") config.pressure.relative_tolerance = std::stof(argv[++i]);
        else if (option == "--divergence-tolerance") config.pressure.divergence_tolerance = std::stof(argv[++i]);
        else if (option == "--max-iterations") config.pressure.max_iterations = std::stoi(argv[++i]);
        else if (option == "--cold") cold = true;
        else if (option == "--profile") profiling = true;
        else if (option == "--audit") audit = true;
        else if (option == "--help") {
            std::cout << "pyro_sparse_benchmark --mode pressure|matrix|coefficients|pipeline --scene none|static|moving|closed|disconnected|thin|opening\n"
                         " --cell-size H --levels L --sources N --source-speed V --age N --samples N --warmup N --duration DT\n"
                         " --checkpoint-in PATH --checkpoint-out PATH --geometry PATH --amplitude A --profile --audit\n"
                         " --initial-blocks N --cold --relative-tolerance E --divergence-tolerance E --max-iterations N\n";
            return 0;
        } else throw std::runtime_error("Unknown option: " + option);
    }
    pyro::sparse::Solver solver(config);
    auto add_box = [&](float3 extent, float3 position) {
        int shape = static_cast<int>(solver.scene.shapes.size()), instance = static_cast<int>(solver.scene.instances.size());
        solver.scene.shapes.push_back({.kind = geometry::ShapeKind::box, .box = {extent}});
        geometry::Instance body{.shape = shape};
        body.motion.rigid.translation[0] = position.x;
        body.motion.rigid.translation[1] = position.y;
        body.motion.rigid.translation[2] = position.z;
        solver.scene.instances.push_back(body);
        solver.colliders.push_back({.instance = instance});
    };
    if (scene == "static" || scene == "moving") {
        solver.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {.5f}});
        geometry::Instance body{.shape = 0};
        body.motion.rigid.translation[2] = 1.5f;
        if (scene == "moving") {
            body.motion.rigid.linear_velocity[0] = .15f;
            body.motion.rigid.angular_velocity[1] = .3f;
        }
        solver.scene.instances.push_back(body);
        solver.colliders.push_back({.instance = 0});
    } else if (scene == "closed" || scene == "disconnected" || scene == "opening") {
        for (int axis = 0; axis < 3; ++axis)
            for (int sign : {-1, 1}) {
                float3 extent{.85f, .85f, .85f}, position{0, 0, .8f};
                if (axis == 0) { extent.x = .06f; position.x = sign * .8f; }
                if (axis == 1) { extent.y = .06f; position.y = sign * .8f; }
                if (axis == 2) { extent.z = .06f; position.z += sign * .8f; }
                add_box(extent, position);
            }
        if (scene == "disconnected") add_box({.06f, .85f, .85f}, {0, 0, .8f});
    } else if (scene == "thin") add_box({.018f, .8f, .8f}, {.3f, 0, .8f});
    else if (scene != "none") throw std::runtime_error("Unknown scene: " + scene);
    int shape = static_cast<int>(solver.scene.shapes.size()), instance = static_cast<int>(solver.scene.instances.size());
    solver.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {.2f}});
    geometry::Instance emitter{.shape = shape};
    emitter.motion.rigid.translation[2] = .25f;
    emitter.motion.rigid.linear_velocity[0] = source_speed;
    solver.scene.instances.push_back(emitter);
    solver.sources.push_back({.instance = instance, .velocity = {0, 0, 1.5f}, .smoke_rate = 1, .temperature_rate = 3, .inherit_velocity = source_speed != 0 ? 1.f : 0.f});
    if (!geometry_path.empty()) {
        std::ifstream input(geometry_path, std::ios::binary);
        input.exceptions(std::ios::badbit | std::ios::failbit);
        for (int index : {0, shape}) {
            int vertices, triangles;
            input.read(reinterpret_cast<char*>(&vertices), sizeof(int));
            input.read(reinterpret_cast<char*>(&triangles), sizeof(int));
            std::vector<float3> points(vertices);
            std::vector<int3> faces(triangles);
            input.read(reinterpret_cast<char*>(points.data()), points.size() * sizeof(float3));
            input.read(reinterpret_cast<char*>(faces.data()), faces.size() * sizeof(int3));
            int asset = solver.scene.add_mesh(points, faces);
            solver.scene.shapes[index] = solver.scene.shapes[asset];
        }
    }
    for (int index = 1; index < source_count; ++index) {
        auto additional = emitter;
        additional.motion.rigid.translation[0] = .5f * index;
        int id = static_cast<int>(solver.scene.instances.size());
        solver.scene.instances.push_back(additional);
        auto source = solver.sources[0];
        source.instance = id;
        solver.sources.push_back(source);
    }
    if (!checkpoint_in.empty()) {
        solver.checkpoint(checkpoint_in.c_str(), true);
        solver.scene.prepare(static_cast<float>(solver.time), static_cast<float>(solver.time));
        std::vector<pyro::ColliderView> views;
        for (auto body : solver.colliders) views.push_back({solver.scene.prepared[body.instance], body.surface_offset});
        auto bodies = ::cuda::make_buffer<pyro::ColliderView>(solver.stream, solver.pool.as_ref(), views.size(), ::cuda::no_init);
        if (!views.empty()) ::cuda::copy_bytes(solver.stream, ::cuda::std::span(views.data(), views.size()), bodies);
        pyro::sparse::sample_geometry(solver.grid, {}, nullptr, bodies.data(), static_cast<int>(views.size()), static_cast<float>(solver.time), 0, solver.stream);
        pyro::sparse::update_geometry(solver.grid, bodies.data(), static_cast<int>(views.size()), static_cast<float>(solver.time), 0, solver.stream);
        solver.synchronize();
    } else if (!cold) solver.advance(.001);
    for (int frame = 0; frame < age; ++frame) solver.advance(duration);
    std::size_t faces = solver.grid.layout.count * 1728ull;
    auto original = ::cuda::make_buffer<float>(solver.stream, solver.pool.as_ref(), mode == "pressure" ? faces : 0, ::cuda::no_init);
    if (mode == "pressure") {
        ::cuda::launch(solver.stream, ::cuda::distribute<256>(faces), seed_velocity, solver.grid, amplitude);
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(solver.grid.state.velocity, faces), original);
        solver.pressure.build(solver.grid, solver.scratch);
    } else if (mode != "matrix" && mode != "coefficients" && mode != "pipeline") throw std::runtime_error("Unknown mode: " + mode);
    ::cuda::timed_event start(solver.device), finish(solver.device);
    std::cout << std::setprecision(10);
    std::cout << "mode,scene,h,amr_levels,sample,time,wall_ms,gpu_ms,iterations,true_residual,divergence,fluid_cells,blocks,capacity,mg_levels,used_bytes,topology_ms,geometry_ms,advection_ms,forces_ms,projection_ms,statistics_ms,substeps,smoke_integral,temperature_integral,peak_used_bytes,projections,matrix_builds,topology_builds,graph_builds\n";
    for (int sample = -warmup; sample < samples; ++sample) {
        if (mode == "pipeline" && scene == "opening") {
            if (sample == 4) solver.colliders.pop_back();
            if (sample == 8) solver.colliders.push_back({.instance = 5});
        }
        std::array<int, 4> counters{solver.pressure.projection_count, solver.pressure.matrix_build_count, solver.pressure.topology_build_count, solver.pressure.graph_build_count};
        if (mode == "pressure") ::cuda::copy_bytes(solver.stream, original, ::cuda::std::span(solver.grid.state.velocity, faces));
        solver.synchronize();
        auto begin = std::chrono::steady_clock::now();
        start.record(solver.stream);
        int iterations = 0;
        if (mode == "pressure") {
            solver.pressure.solution_valid = false;
            iterations = solver.pressure.project(solver.grid, solver.grid.state, solver.scratch);
        }
        else if (mode == "matrix" || mode == "coefficients") {
            if (mode == "matrix") solver.pressure.topology_dirty = true;
            solver.pressure.build(solver.grid, solver.scratch);
        }
        else {
            if (profiling) solver.advance<true>(duration);
            else solver.advance(duration);
            iterations = solver.statistics.pressure_iterations;
        }
        finish.record(solver.stream);
        solver.synchronize();
        float wall_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        float gpu_ms = static_cast<float>((finish - start).count()) * 1.e-6f;
        if (sample < 0) continue;
        pyro::sparse::IterationState state{};
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(solver.pressure.view.iteration, 1), ::cuda::std::span(&state, 1), {.src_access_order = ::cuda::source_access_order::stream});
        pyro::sparse::measure(solver.grid, config.forces, nullptr, 0, static_cast<float>(solver.time), true, solver.pressure.view.reductions, solver.stream);
        std::array<double, 6> measurements{};
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(solver.pressure.view.reductions + solver.grid.layout.count * 6ull, 6), ::cuda::std::span(measurements.data(), measurements.size()), {.src_access_order = ::cuda::source_access_order::stream});
        solver.synchronize();
        if (!std::isfinite(measurements[1]) || (mode != "matrix" && mode != "coefficients" && measurements[1] > config.pressure.divergence_tolerance)) throw std::runtime_error("Projection divergence exceeds benchmark accuracy requirement: " + std::to_string(measurements[1]));
        const auto& stats = solver.statistics;
        std::cout << mode << ',' << scene << ',' << config.grid.cell_size << ',' << config.grid.levels << ',' << sample << ',' << solver.time << ',' << wall_ms << ',' << gpu_ms << ',' << iterations << ',' << state.max_residual << ',' << measurements[1] << ',' << measurements[0] << ',' << solver.grid.layout.count << ',' << solver.grid.layout.capacity << ',' << solver.pressure.view.level_count << ',' << solver.pool.attribute(::cuda::memory_pool_attributes::used_mem_current) << ',' << stats.topology_milliseconds << ',' << stats.geometry_milliseconds << ',' << stats.advection_milliseconds << ',' << stats.forces_milliseconds << ',' << stats.projection_milliseconds << ',' << stats.statistics_milliseconds << ',' << stats.substeps << ',' << measurements[3] << ',' << measurements[4] << ',' << solver.pool.attribute(::cuda::memory_pool_attributes::used_mem_high) << ',' << solver.pressure.projection_count - counters[0] << ',' << solver.pressure.matrix_build_count - counters[1] << ',' << solver.pressure.topology_build_count - counters[2] << ',' << solver.pressure.graph_build_count - counters[3] << '\n';
    }
    if (audit) {
        auto download = [&]<typename T>(T* pointer, std::size_t count) {
            std::vector<T> data(count);
            ::cuda::copy_bytes(solver.stream, ::cuda::std::span(pointer, count), ::cuda::std::span(data.data(), count), {.src_access_order = ::cuda::source_access_order::stream});
            solver.synchronize();
            return data;
        };
        int count = solver.grid.layout.count;
        auto blocks = download(solver.grid.layout.blocks, count);
        auto face_data = download(solver.grid.faces, count * 1728ull);
        auto links = download(solver.grid.links, count * 384ull);
        auto fluid = download(solver.grid.fluid, count * 512ull);
        auto velocity = download(solver.grid.state.velocity, count * 1728ull);
        auto solid = download(solver.grid.solid_divergence, count * 512ull);
        double maximum = 0, squared = 0, volume = 0;
        for (int i = 0; i < count * 512; ++i) {
            if (!fluid[i]) continue;
            int local = i & 511, block = i >> 9;
            double h = std::ldexp(double(config.grid.cell_size), blocks[block].level);
            double flux = solver.grid.boundary_mode == pyro::BoundaryMode::moving ? -double(solid[i]) * h * h * h : 0;
            for (int axis = 0; axis < 3; ++axis)
                for (int sign : {-1, 1}) {
                    int coordinate = (local >> (3 * axis)) & 7;
                    std::array<int, 4> adjacent{-1, -1, -1, -1};
                    if ((sign < 0 && coordinate > 0) || (sign > 0 && coordinate < 7)) adjacent[0] = block * 1728 + axis * 576 + local + (sign > 0 ? 1 << (3 * axis) : 0);
                    else {
                        int transverse = axis == 0 ? ((local >> 3) & 7) + 8 * (local >> 6) : axis == 1 ? (local & 7) + 8 * (local >> 6) : (local & 7) + (local & 56);
                        auto group = links[block * 384 + (axis * 2 + (sign > 0)) * 64 + transverse];
                        for (int q = 0; q < 4; ++q) adjacent[q] = group.face[q];
                    }
                    for (int f : adjacent)
                        if (f >= 0) flux += sign * (double(face_data[f].area) * velocity[f] + (solver.grid.boundary_mode == pyro::BoundaryMode::moving ? face_data[f].solid_flux : 0));
                }
            double divergence = flux / (h * h * h);
            maximum = std::max(maximum, std::isfinite(divergence) ? std::abs(divergence) : INFINITY);
            squared += divergence * divergence * h * h * h;
            volume += h * h * h;
        }
        std::cerr << std::setprecision(12) << "{\"cpu_max_divergence\":" << maximum << ",\"volume_l2_divergence\":" << std::sqrt(squared / volume) << ",\"cells\":" << count * 512 << "}\n";
        if (!std::isfinite(maximum) || maximum > config.pressure.divergence_tolerance) throw std::runtime_error("CPU projection audit failed");

        auto descriptors = download(solver.pressure.view.levels, solver.pressure.view.level_count);
        std::vector<pyro::sparse::Edge> previous_edges;
        std::vector<float> previous_boundary;
        auto apply = [](const auto& edges, const auto& boundary, const std::vector<double>& x) {
            std::vector<double> result(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) result[i] = boundary[i] * x[i];
            for (auto edge : edges)
                if (edge.weight != 0 && edge.left >= 0 && edge.right >= 0) {
                    double value = edge.weight * (x[edge.left] - x[edge.right]);
                    result[edge.left] += value;
                    result[edge.right] -= value;
                }
            return result;
        };
        auto relative_error = [](const auto& a, const auto& b) {
            double error = 0, norm = 1;
            for (std::size_t i = 0; i < a.size(); ++i) {
                error = std::max(error, std::abs(a[i] - b[i]));
                norm = std::max(norm, std::abs(a[i]));
            }
            return error / norm;
        };
        auto anchors = solver.pressure.view.closed ? download(solver.pressure.view.anchors, count * 512ull) : std::vector<unsigned char>(count * 512ull);
        for (int l = 0; l < static_cast<int>(descriptors.size()); ++l) {
            auto level = descriptors[l];
            auto edges = download(level.edges, level.edge_count);
            auto boundary = download(level.boundary, level.count);
            std::vector<double> x(level.count), y(level.count);
            for (int i = 0; i < level.count; ++i) {
                x[i] = std::sin((i + 1) * 1.23456789);
                y[i] = std::cos((i + 1) * .87654321);
                if (l == 0 && (!fluid[i] || anchors[i])) x[i] = y[i] = 0;
            }
            auto ax = apply(edges, boundary, x), ay = apply(edges, boundary, y);
            double energy_x = 0, energy_y = 0, xy = 0, yx = 0, galerkin_error = 0, flux_error = 0, coarse_scale = 0;
            for (int i = 0; i < level.count; ++i) {
                energy_x += x[i] * ax[i];
                energy_y += y[i] * ay[i];
                xy += x[i] * ay[i];
                yx += y[i] * ax[i];
            }
            if (l == 0) {
                std::vector<double> physical(level.count);
                for (int f = 0; f < count * 1728; ++f) {
                    auto face = face_data[f];
                    if (face.weight <= 0) continue;
                    double difference = (face.left >= 0 ? x[face.left] : 0) - (face.right >= 0 ? x[face.right] : 0);
                    if (face.group >= 0) {
                        double average = 0, total = 0;
                        for (int peer : links[face.group].face) {
                            auto adjacent = face_data[peer];
                            int fine = blocks[adjacent.left >> 9].level < blocks[adjacent.right >> 9].level ? adjacent.left : adjacent.right;
                            average += adjacent.weight * x[fine];
                            total += adjacent.weight;
                        }
                        average /= total;
                        difference = blocks[face.left >> 9].level < blocks[face.right >> 9].level ? average - x[face.right] : x[face.left] - average;
                    }
                    double value = double(face.area) * difference / (double(face.distance) * config.grid.cell_size);
                    if (face.left >= 0) physical[face.left] += value;
                    if (face.right >= 0) physical[face.right] -= value;
                }
                for (int i = 0; i < level.count; ++i) if (anchors[i]) physical[i] = x[i];
                flux_error = relative_error(ax, physical);
            } else {
                auto parents = download(descriptors[l - 1].parent, descriptors[l - 1].count);
                std::vector<double> prolonged(parents.size());
                for (std::size_t i = 0; i < parents.size(); ++i) prolonged[i] = x[parents[i]];
                auto previous_product = apply(previous_edges, previous_boundary, prolonged);
                std::vector<double> expected(level.count);
                for (std::size_t i = 0; i < parents.size(); ++i) expected[parents[i]] += previous_product[i];
                double cross = 0, norm = 0;
                for (int i = 0; i < level.count; ++i) {
                    cross += ax[i] * expected[i];
                    norm += expected[i] * expected[i];
                }
                coarse_scale = cross / norm;
                for (double& value : expected) value *= coarse_scale;
                galerkin_error = relative_error(ax, expected);
            }
            double symmetry_error = std::abs(xy - yx) / std::max(1.0, std::sqrt(energy_x * energy_y));
            std::cerr << "{\"operator_level\":" << l << ",\"nodes\":" << level.count << ",\"edge_slots\":" << level.edge_count << ",\"energy\":" << energy_x << ",\"symmetry_error\":" << symmetry_error << ",\"coarse_scale\":" << coarse_scale << ",\"coarse_operator_error\":" << galerkin_error << ",\"flux_operator_error\":" << flux_error << "}\n";
            if (!(energy_x > 0) || !(energy_y > 0) || symmetry_error > 1.e-10 || galerkin_error > 3.e-6 || flux_error > 3.e-6) throw std::runtime_error("CPU operator audit failed");
            previous_edges = std::move(edges);
            previous_boundary = std::move(boundary);
        }

    }
    if (!checkpoint_out.empty()) solver.checkpoint(checkpoint_out.c_str(), false);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
