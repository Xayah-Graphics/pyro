#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>
#include <cuda_profiler_api.h>
#include <solver/dense/advection.cuh>
#include <solver/dense/solver.cuh>
#include <solver/sparse/advection.cuh>
#include <solver/sparse/solver.cuh>

__host__ __device__ float tracer(float3 p) {
    float x = p.x - .25f, y = p.y, z = p.z - .1f;
    return expf(-60 * (x * x + y * y + z * z));
}

__host__ __device__ float3 velocity(float3 p, int experiment) {
    if (experiment == 0) return {.15f, -.08f, .05f};
    if (experiment == 1) return {-p.y, p.x, 0};
    return {.2f * sinf(3.141592654f * p.x) * cosf(3.141592654f * p.y), -.2f * cosf(3.141592654f * p.x) * sinf(3.141592654f * p.y), .03f * sinf(3.141592654f * p.x)};
}

template <bool sparse, typename Grid>
__global__ void seed(Grid grid, float h, int experiment) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (sparse) {
        if (i < grid.layout.count * 512) {
            float q = grid.fluid[i] ? tracer(pyro::sparse::center(grid.layout, i)) : 0;
            grid.state.smoke[i] = q;
            grid.state.temperature[i] = .7f * q;
        }
        if (i < grid.layout.count * 1728) {
            float3 p;
            float spacing;
            int axis, owner, sign;
            pyro::sparse::face_position(grid.layout, i, p, spacing, axis, owner, sign);
            float3 v = velocity(p, experiment);
            grid.state.velocity[i] = grid.faces[i].area > 0 ? (axis == 0 ? v.x : axis == 1 ? v.y : v.z) : grid.faces[i].solid_flux / (spacing * spacing);
        }
    } else {
        if (i >= grid.layout.cells) return;
        int3 c = pyro::dense::cell_coordinate(grid.layout, i);
        float3 p{grid.origin.x + (c.x + .5f) * h, grid.origin.y + (c.y + .5f) * h, grid.origin.z + (c.z + .5f) * h};
        float q = grid.fluid[i] ? tracer(p) : 0;
        grid.state.smoke[i] = q;
        grid.state.temperature[i] = .7f * q;
        for (int axis = 0; axis < 3; ++axis) {
            float3 point{p.x - (axis == 0 ? .5f * h : 0), p.y - (axis == 1 ? .5f * h : 0), p.z - (axis == 2 ? .5f * h : 0)};
            float3 v = velocity(point, experiment);
            grid.state.velocity[axis][i] = grid.aperture[axis][i] > 0 ? (axis == 0 ? v.x : axis == 1 ? v.y : v.z) : grid.solid_velocity[axis][i];
        }
    }
}

int main(int argc, char** argv) try {
    std::string backend = "dense", experiment = "rotation", mode = "frozen", scene = "none", input_path, output_path, geometry_path;
    int3 resolution{128, 128, 128};
    int levels = 3, samples = 20, warmup = 3;
    float h = .025f, dt = .02f;
    bool midpoint = true, profiling = false;
    for (int i = 1; i < argc; ++i) {
        std::string option = argv[i];
        if (option == "--backend") backend = argv[++i];
        else if (option == "--case") experiment = argv[++i];
        else if (option == "--mode") mode = argv[++i];
        else if (option == "--scene") scene = argv[++i];
        else if (option == "--resolution") { resolution.x = std::stoi(argv[++i]); resolution.y = std::stoi(argv[++i]); resolution.z = std::stoi(argv[++i]); }
        else if (option == "--levels") levels = std::stoi(argv[++i]);
        else if (option == "--cell-size") h = std::stof(argv[++i]);
        else if (option == "--dt") dt = std::stof(argv[++i]);
        else if (option == "--samples") samples = std::stoi(argv[++i]);
        else if (option == "--warmup") warmup = std::stoi(argv[++i]);
        else if (option == "--checkpoint-in") input_path = argv[++i];
        else if (option == "--dump") output_path = argv[++i];
        else if (option == "--geometry") geometry_path = argv[++i];
        else if (option == "--carrier-state") midpoint = false;
        else if (option == "--profile") profiling = true;
        else if (option == "--help") {
            std::cout << "pyro_advection_benchmark --backend dense|sparse --case translation|rotation|vortex|checkpoint --mode frozen|sequence\n"
                         " --scene none|static|moving --resolution NX NY NZ --cell-size H --levels N --samples N --warmup N --dt DT\n"
                         " --checkpoint-in PATH --geometry PATH --dump PATH --carrier-state --profile\n";
            return 0;
        } else throw std::runtime_error("Unknown option: " + option);
    }
    int kind = experiment == "translation" ? 0 : experiment == "rotation" ? 1 : 2;
    float3 wind = kind == 0 ? velocity({}, 0) : float3{};
    auto run = [&]<bool sparse, typename Solver>(Solver& solver, float spacing) {
        if (scene != "none") {
            solver.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {experiment == "checkpoint" ? .5f : .3f}});
            geometry::Instance body{.shape = 0};
            body.motion.rigid.translation[2] = experiment == "checkpoint" ? 1.5f : .1f;
            if (scene == "moving") { body.motion.rigid.linear_velocity[0] = .1f; body.motion.rigid.angular_velocity[1] = .3f; }
            solver.scene.instances.push_back(body);
            solver.colliders.push_back({.instance = 0});
        }
        if constexpr (sparse) {
            int shape = static_cast<int>(solver.scene.shapes.size()), instance = static_cast<int>(solver.scene.instances.size());
            solver.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {experiment == "checkpoint" ? .2f : .65f}});
            geometry::Instance emitter{.shape = shape};
            emitter.motion.rigid.translation[2] = experiment == "checkpoint" ? .25f : 0;
            solver.scene.instances.push_back(emitter);
            solver.sources.push_back({.instance = instance, .smoke_rate = 1});
            if (!geometry_path.empty()) {
                std::ifstream input(geometry_path, std::ios::binary);
                input.exceptions(std::ios::badbit | std::ios::failbit);
                for (int index : {0, shape}) {
                    int vertices, triangles;
                    input.read(reinterpret_cast<char*>(&vertices), 4);
                    input.read(reinterpret_cast<char*>(&triangles), 4);
                    std::vector<float3> points(vertices);
                    std::vector<int3> faces(triangles);
                    input.read(reinterpret_cast<char*>(points.data()), points.size() * sizeof(float3));
                    input.read(reinterpret_cast<char*>(faces.data()), faces.size() * sizeof(int3));
                    int asset = solver.scene.add_mesh(points, faces);
                    solver.scene.shapes[index] = solver.scene.shapes[asset];
                }
            }
            if (!input_path.empty()) solver.checkpoint(input_path.c_str(), true);
            else solver.advance(.001);
        } else {
            if (!input_path.empty()) {
                std::ifstream input(input_path, std::ios::binary);
                input.exceptions(std::ios::badbit | std::ios::failbit);
                std::array<int, 6> header;
                input.read(reinterpret_cast<char*>(header.data()), sizeof(header));
                std::vector<float> host(solver.grid.layout.cells);
                for (auto* field : {solver.grid.state.velocity[0], solver.grid.state.velocity[1], solver.grid.state.velocity[2], solver.grid.state.smoke, solver.grid.state.temperature}) {
                    input.read(reinterpret_cast<char*>(host.data()), host.size() * sizeof(float));
                    ::cuda::copy_bytes(solver.stream, ::cuda::std::span(host.data(), host.size()), ::cuda::std::span(field, host.size()), {.src_access_order = ::cuda::source_access_order::during_api_call});
                }
            }
        }
        auto& grid = solver.grid;
        solver.scene.prepare(0, dt);
        if constexpr (!sparse) {
            float3 low{1.e30f, 1.e30f, 1.e30f}, high{-1.e30f, -1.e30f, -1.e30f};
            for (auto body : solver.colliders) {
                const auto& bounds = solver.scene.prepared[body.instance].swept;
                float padding = std::abs(body.surface_offset) + 4 * spacing;
                low = {std::min(low.x, bounds.lower.x - padding), std::min(low.y, bounds.lower.y - padding), std::min(low.z, bounds.lower.z - padding)};
                high = {std::max(high.x, bounds.upper.x + padding), std::max(high.y, bounds.upper.y + padding), std::max(high.z, bounds.upper.z + padding)};
            }
            if (!solver.colliders.empty()) {
                grid.geometry_lower = {std::max(-1, int(std::floor((low.x - grid.origin.x) / spacing)) - 1), std::max(-1, int(std::floor((low.y - grid.origin.y) / spacing)) - 1), std::max(-1, int(std::floor((low.z - grid.origin.z) / spacing)) - 1)};
                int3 upper{std::min(grid.layout.size.x, int(std::ceil((high.x - grid.origin.x) / spacing)) + 1), std::min(grid.layout.size.y, int(std::ceil((high.y - grid.origin.y) / spacing)) + 1), std::min(grid.layout.size.z, int(std::ceil((high.z - grid.origin.z) / spacing)) + 1)};
                grid.geometry_extent = {upper.x - grid.geometry_lower.x + 1, upper.y - grid.geometry_lower.y + 1, upper.z - grid.geometry_lower.z + 1};
            }
        }
        std::vector<pyro::ColliderView> colliders;
        grid.collider_speed = 0;
        for (auto body : solver.colliders) {
            colliders.push_back({solver.scene.prepared[body.instance], body.surface_offset});
            grid.collider_speed = std::max(grid.collider_speed, solver.scene.prepared[body.instance].max_speed);
        }
        auto bodies = ::cuda::make_buffer<pyro::ColliderView>(solver.stream, solver.pool.as_ref(), colliders.size(), ::cuda::no_init);
        if (!colliders.empty()) ::cuda::copy_bytes(solver.stream, ::cuda::std::span(colliders.data(), colliders.size()), bodies, {.src_access_order = ::cuda::source_access_order::during_api_call});
        grid.geometry_time = 0;
        grid.geometry_duration = dt;
        grid.boundary_mode = scene == "none" ? pyro::BoundaryMode::none : scene == "moving" ? pyro::BoundaryMode::moving : pyro::BoundaryMode::stationary;
        for (int sample = 0; sample < 3; ++sample) {
            if constexpr (sparse) pyro::sparse::sample_geometry(grid, {}, nullptr, bodies.data(), static_cast<int>(colliders.size()), sample * .5f * dt, sample, solver.stream);
            else pyro::dense::sample_geometry(grid, spacing, bodies.data(), static_cast<int>(colliders.size()), sample * .5f * dt, sample, solver.stream);
        }
        if constexpr (sparse) pyro::sparse::update_geometry(grid, bodies.data(), static_cast<int>(colliders.size()), 0, 0, solver.stream);
        else pyro::dense::update_boundary(grid, spacing, bodies.data(), static_cast<int>(colliders.size()), 0, 0, solver.stream);
        std::size_t cells, faces;
        if constexpr (sparse) { cells = grid.layout.count * 512ull; faces = grid.layout.count * 1728ull; }
        else { cells = grid.layout.cells; faces = cells * 3; }
        if (experiment != "checkpoint") ::cuda::launch(solver.stream, ::cuda::distribute<256>(sparse ? faces : cells), seed<sparse, decltype(solver.grid)>, grid, spacing, kind);
        if constexpr (sparse) {
            ::cuda::copy_bytes(solver.stream, ::cuda::std::span(grid.state.velocity, faces), ::cuda::std::span(grid.midpoint.velocity, faces));
        } else {
            for (int axis = 0; axis < 3; ++axis) ::cuda::copy_bytes(solver.stream, ::cuda::std::span(grid.state.velocity[axis], cells), ::cuda::std::span(grid.midpoint.velocity[axis], cells));
        }
        auto carrier = midpoint ? grid.midpoint : grid.state;
        ::cuda::timed_event start(solver.device), finish(solver.device);
        solver.synchronize();
        if (profiling) cudaProfilerStart();
        std::cout << std::setprecision(10) << "backend,case,scene,mode,sample,cells,wall_ms,gpu_ms,used_bytes\n";
        for (int sample = mode == "sequence" ? 0 : -warmup; sample < samples; ++sample) {
            solver.synchronize();
            auto begin = std::chrono::steady_clock::now();
            start.record(solver.stream);
            if constexpr (sparse) pyro::sparse::advect(grid, grid.next, carrier, spacing, wind, 0, dt, solver.stream);
            else pyro::dense::advect(grid, spacing, wind, grid.next, grid.state, carrier, 0, dt, solver.stream);
            finish.record(solver.stream);
            solver.synchronize();
            float wall = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
            float gpu = static_cast<float>((finish - start).count()) * 1.e-6f;
            if (sample >= 0) std::cout << backend << ',' << experiment << ',' << scene << ',' << mode << ',' << sample << ',' << cells << ',' << wall << ',' << gpu << ',' << solver.pool.attribute(::cuda::memory_pool_attributes::used_mem_current) << '\n';
            if (mode == "sequence") std::swap(grid.state, grid.next);
        }
        if (profiling) cudaProfilerStop();
        auto output = mode == "sequence" ? grid.state : grid.next;
        std::vector<float> smoke(cells), temperature(cells);
        std::vector<unsigned char> fluid(cells);
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(output.smoke, cells), ::cuda::std::span(smoke.data(), cells), {.src_access_order = ::cuda::source_access_order::stream});
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(output.temperature, cells), ::cuda::std::span(temperature.data(), cells), {.src_access_order = ::cuda::source_access_order::stream});
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(grid.fluid, cells), ::cuda::std::span(fluid.data(), cells), {.src_access_order = ::cuda::source_access_order::stream});
        std::vector<pyro::sparse::Block> blocks;
        if constexpr (sparse) {
            blocks.resize(grid.layout.count);
            ::cuda::copy_bytes(solver.stream, ::cuda::std::span(grid.layout.blocks, blocks.size()), ::cuda::std::span(blocks.data(), blocks.size()), {.src_access_order = ::cuda::source_access_order::stream});
        }
        solver.synchronize();
        double square = 0, norm = 0, mass = 0, reference_mass = 0, minimum = 1e30, maximum = -1e30;
        double elapsed = dt * (mode == "sequence" ? samples : 1);
        for (std::size_t i = 0; i < cells; ++i) {
            if (!fluid[i]) continue;
            float3 point;
            float cell_h = spacing;
            if constexpr (sparse) {
                const auto& block = blocks[i / 512];
                cell_h = std::ldexp(spacing, block.level);
                int local = static_cast<int>(i & 511);
                point = {(block.coordinate.x * 8 + (local & 7) + .5f) * cell_h, (block.coordinate.y * 8 + ((local >> 3) & 7) + .5f) * cell_h, (block.coordinate.z * 8 + ((local >> 6) & 7) + .5f) * cell_h};
            } else {
                int z = i / grid.layout.slice, r = i - z * grid.layout.slice, y = r / grid.layout.pitch, x = r - y * grid.layout.pitch;
                if (x < 1 || y < 1 || z < 1 || x > grid.layout.size.x || y > grid.layout.size.y || z > grid.layout.size.z) continue;
                point = {grid.origin.x + (x - .5f) * cell_h, grid.origin.y + (y - .5f) * cell_h, grid.origin.z + (z - .5f) * cell_h};
            }
            float3 departure = point;
            if (kind == 0) { departure.x -= .15f * elapsed; departure.y += .08f * elapsed; departure.z -= .05f * elapsed; }
            if (kind == 1) { departure.x = point.x * std::cos(elapsed) + point.y * std::sin(elapsed); departure.y = -point.x * std::sin(elapsed) + point.y * std::cos(elapsed); }
            double reference = tracer(departure), volume = double(cell_h) * cell_h * cell_h;
            square += (smoke[i] - reference) * (smoke[i] - reference) * volume;
            norm += reference * reference * volume;
            mass += smoke[i] * volume;
            reference_mass += reference * volume;
            minimum = std::min(minimum, double(smoke[i])); maximum = std::max(maximum, double(smoke[i]));
            if (!std::isfinite(smoke[i]) || !std::isfinite(temperature[i])) throw std::runtime_error("Nonfinite transported scalar");
        }
        bool analytic = kind < 2 && scene == "none" && experiment != "checkpoint";
        std::cerr << std::setprecision(12) << "{\"relative_l2\":";
        if (analytic) std::cerr << std::sqrt(square / norm);
        else std::cerr << "null";
        std::cerr << ",\"mass\":" << mass << ",\"reference_mass\":";
        if (analytic) std::cerr << reference_mass;
        else std::cerr << "null";
        std::cerr << ",\"minimum\":" << minimum << ",\"maximum\":" << maximum << ",\"analytic_reference\":" << (analytic ? "true" : "false") << "}\n";
        if (!output_path.empty()) {
            std::ofstream dump(output_path, std::ios::binary);
            dump.exceptions(std::ios::badbit | std::ios::failbit);
            std::array<unsigned long long, 4> header{sparse ? 1ull : 0ull, cells, faces, blocks.size()};
            dump.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
            dump.write(reinterpret_cast<const char*>(blocks.data()), blocks.size() * sizeof(pyro::sparse::Block));
            dump.write(reinterpret_cast<const char*>(smoke.data()), cells * sizeof(float));
            dump.write(reinterpret_cast<const char*>(temperature.data()), cells * sizeof(float));
            std::vector<float> host(sparse ? faces : cells);
            if constexpr (sparse) {
                ::cuda::copy_bytes(solver.stream, ::cuda::std::span(output.velocity, faces), ::cuda::std::span(host.data(), faces), {.src_access_order = ::cuda::source_access_order::stream});
                solver.synchronize(); dump.write(reinterpret_cast<const char*>(host.data()), faces * sizeof(float));
            } else {
                for (int axis = 0; axis < 3; ++axis) {
                    ::cuda::copy_bytes(solver.stream, ::cuda::std::span(output.velocity[axis], cells), ::cuda::std::span(host.data(), cells), {.src_access_order = ::cuda::source_access_order::stream});
                    solver.synchronize(); dump.write(reinterpret_cast<const char*>(host.data()), cells * sizeof(float));
                }
            }
        }
    };
    if (backend == "sparse") {
        pyro::sparse::Config config;
        config.grid.cell_size = h; config.grid.levels = levels; config.forces.up = {}; config.grid.wind = wind;
        pyro::sparse::Solver solver(config);
        run.template operator()<true>(solver, h);
    } else {
        pyro::dense::Config config;
        config.resolution = resolution; config.cell_size = 2.f / std::min({resolution.x, resolution.y, resolution.z}); config.wind = wind; config.forces.up = {};
        config.origin = {-.5f * resolution.x * config.cell_size, -.5f * resolution.y * config.cell_size, -.5f * resolution.z * config.cell_size};
        pyro::dense::Solver solver(config);
        run.template operator()<false>(solver, config.cell_size);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
