#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>
#include <solver/dense/solver.cuh>

__global__ void seed_velocity(pyro::dense::Grid grid, float amplitude) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grid.layout.cells) return;
    int3 p = pyro::dense::cell_coordinate(grid.layout, i);
    for (int axis = 0; axis < 3; ++axis) {
        if (!pyro::dense::face_valid(grid.layout, p, axis) || grid.aperture[axis][i] <= 0) continue;
        float x        = (p.x + (axis != 0 ? .5f : 0.f)) / grid.layout.size.x;
        float y        = (p.y + (axis != 1 ? .5f : 0.f)) / grid.layout.size.y;
        float z        = (p.z + (axis != 2 ? .5f : 0.f)) / grid.layout.size.z;
        float a        = axis == 0 ? x : axis == 1 ? y : z;
        float b        = axis == 0 ? y : axis == 1 ? z : x;
        float c        = axis == 0 ? z : axis == 1 ? x : y;
        float velocity = sinf(6.283185307f * a) * cosf(3.141592654f * b) * cosf(3.141592654f * c);
        velocity += .2f * sinf(25.13274123f * a) * sinf(18.84955592f * b) * cosf(12.56637061f * c);
        grid.state.velocity[axis][i] = amplitude * velocity;
    }
}

int main(int argc, char** argv) try {
    pyro::dense::Config config;
    config.resolution = {128, 128, 128};
    config.forces.up  = {0, 0, 1};
    std::string scene = "static", mode = "pressure", snapshot;
    int samples = 12, warmup = 3;
    double duration = 1.0 / 24;
    float amplitude = .03f;
    bool profiling = false, audit = false;
    for (int i = 1; i < argc; ++i) {
        std::string option = argv[i];
        if (option == "--resolution") {
            config.resolution.x = std::stoi(argv[++i]);
            config.resolution.y = std::stoi(argv[++i]);
            config.resolution.z = std::stoi(argv[++i]);
        } else if (option == "--scene") scene = argv[++i];
        else if (option == "--mode") mode = argv[++i];
        else if (option == "--samples") samples = std::stoi(argv[++i]);
        else if (option == "--warmup") warmup = std::stoi(argv[++i]);
        else if (option == "--snapshot") snapshot = argv[++i];
        else if (option == "--amplitude") amplitude = std::stof(argv[++i]);
        else if (option == "--duration") duration = std::stod(argv[++i]);
        else if (option == "--relative-tolerance") config.pressure.relative_tolerance = std::stof(argv[++i]);
        else if (option == "--divergence-tolerance") config.pressure.divergence_tolerance = std::stof(argv[++i]);
        else if (option == "--max-iterations") config.pressure.max_iterations = std::stoi(argv[++i]);
        else if (option == "--profile") profiling = true;
        else if (option == "--audit") audit = true;
        else if (option == "--help") {
            std::cout << "pyro_dense_benchmark --mode pressure|matrix|pipeline --scene none|static|moving|closed|disconnected|thin\n"
                         "                     --resolution NX NY NZ --warmup N --samples N --profile --audit\n"
                         "                     --snapshot PATH --amplitude A --duration DT\n"
                         "                     --relative-tolerance E --divergence-tolerance E --max-iterations N\n";
            return 0;
        } else throw std::runtime_error("Unknown option: " + option);
    }
    config.cell_size = 2.f / std::min({config.resolution.x, config.resolution.y, config.resolution.z});
    config.origin    = {-config.resolution.x * config.cell_size * .5f, -config.resolution.y * config.cell_size * .5f, -config.resolution.z * config.cell_size * .5f};
    pyro::dense::Solver solver(config);
    auto add_box = [&](float3 extent, float3 position, bool moving) {
        int shape = static_cast<int>(solver.scene.shapes.size());
        solver.scene.shapes.push_back({.kind = geometry::ShapeKind::box, .box = {extent}});
        geometry::Instance instance{.shape = shape};
        instance.motion.rigid.translation[0] = position.x;
        instance.motion.rigid.translation[1] = position.y;
        instance.motion.rigid.translation[2] = position.z;
        if (moving) {
            instance.motion.rigid.linear_velocity[0]  = .1f;
            instance.motion.rigid.angular_velocity[1] = .3f;
        }
        int index = static_cast<int>(solver.scene.instances.size());
        solver.scene.instances.push_back(instance);
        solver.colliders.push_back({.instance = index});
    };
    if (scene == "static" || scene == "moving") {
        int shape = static_cast<int>(solver.scene.shapes.size()), instance = static_cast<int>(solver.scene.instances.size());
        solver.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {.3f}});
        geometry::Instance body{.shape = shape};
        body.motion.rigid.translation[2] = .1f;
        if (scene == "moving") body.motion.rigid.linear_velocity[0] = .1f;
        solver.scene.instances.push_back(body);
        solver.colliders.push_back({.instance = instance});
    } else if (scene == "closed" || scene == "disconnected") {
        add_box({.04f, .85f, .85f}, {-.8f, 0, 0}, false);
        add_box({.04f, .85f, .85f}, {.8f, 0, 0}, false);
        add_box({.85f, .04f, .85f}, {0, -.8f, 0}, false);
        add_box({.85f, .04f, .85f}, {0, .8f, 0}, false);
        add_box({.85f, .85f, .04f}, {0, 0, -.8f}, false);
        add_box({.85f, .85f, .04f}, {0, 0, .8f}, false);
        if (scene == "disconnected") add_box({.04f, .85f, .85f}, {0, 0, 0}, false);
    } else if (scene == "thin") {
        add_box({.018f, .9f, .9f}, {0, -.15f, 0}, false);
        add_box({.2f, .25f, .25f}, {.3f, .2f, 0}, false);
    } else if (scene != "none") throw std::runtime_error("Unknown scene: " + scene);
    if (mode == "pipeline") {
        int shape = static_cast<int>(solver.scene.shapes.size()), instance = static_cast<int>(solver.scene.instances.size());
        solver.scene.shapes.push_back({.kind = geometry::ShapeKind::sphere, .sphere = {.15f}});
        geometry::Instance emitter{.shape = shape};
        emitter.motion.rigid.translation[2] = -.65f;
        solver.scene.instances.push_back(emitter);
        solver.sources.push_back({.instance = instance, .velocity = {0, 0, .8f}, .smoke_rate = 1, .temperature_rate = 3});
    }
    solver.advance(.001);
    std::size_t cells = solver.grid.layout.cells;
    std::vector<::cuda::buffer<float, ::cuda::mr::device_accessible>> original;
    std::array<float*, 5> fields{solver.grid.state.velocity[0], solver.grid.state.velocity[1], solver.grid.state.velocity[2], solver.grid.state.smoke, solver.grid.state.temperature};
    if (mode == "pressure") {
        ::cuda::launch(solver.stream, ::cuda::distribute<256>(cells), seed_velocity, solver.grid, amplitude);
        for (auto* field : fields) {
            auto& buffer = original.emplace_back(::cuda::make_buffer<float>(solver.stream, solver.pool.as_ref(), cells, ::cuda::no_init));
            ::cuda::copy_bytes(solver.stream, ::cuda::std::span(field, cells), buffer);
        }
    } else if (mode != "matrix" && mode != "pipeline") throw std::runtime_error("Unknown mode: " + mode);
    ::cuda::timed_event start(solver.device), finish(solver.device);
    auto maximum = ::cuda::make_buffer<float>(solver.stream, solver.pool.as_ref(), 2, ::cuda::no_init);
    std::cout << std::setprecision(10);
    std::cout << "mode,scene,nx,ny,nz,sample,time,wall_ms,gpu_ms,iterations,true_residual,divergence,fluid_cells,levels,used_bytes,geometry_ms,advection_ms,forces_ms,projection_ms,statistics_ms,substeps\n";
    for (int sample = -warmup; sample < samples; ++sample) {
        if (mode == "pressure")
            for (int field = 0; field < 5; ++field) ::cuda::copy_bytes(solver.stream, original[field], ::cuda::std::span(fields[field], cells));
        solver.synchronize();
        auto begin = std::chrono::steady_clock::now();
        start.record(solver.stream);
        int iterations = 0;
        if (mode == "pressure") iterations = solver.pressure.project(solver.grid, solver.grid.state, config.cell_size, config.wind);
        else if (mode == "matrix") solver.pressure.build(solver.grid, config.cell_size);
        else {
            if (profiling) solver.advance<true>(duration);
            else solver.advance(duration);
            iterations = solver.statistics.pressure_iterations;
        }
        finish.record(solver.stream);
        solver.synchronize();
        float wall_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
        float gpu_ms  = static_cast<float>((finish - start).count()) * 1.e-6f;
        if (sample < 0) continue;
        pyro::dense::IterationState state{};
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(solver.pressure.view.iteration, 1), ::cuda::std::span(&state, 1), {.src_access_order = ::cuda::source_access_order::stream});
        pyro::dense::measure(solver.grid, solver.grid.state, config.forces, config.cell_size, config.wind, config.max_time_step, nullptr, 0, static_cast<float>(solver.time), solver.pressure.view.reductions, maximum.data(), true, solver.stream);
        std::array<double, 6> measurements{};
        ::cuda::copy_bytes(solver.stream, ::cuda::std::span(solver.pressure.view.reductions + solver.grid.layout.blocks * 6ull, 6), ::cuda::std::span(measurements.data(), measurements.size()), {.src_access_order = ::cuda::source_access_order::stream});
        solver.synchronize();
        if (!std::isfinite(measurements[1]) || (mode != "matrix" && measurements[1] > config.pressure.divergence_tolerance)) throw std::runtime_error("Projection divergence exceeds benchmark accuracy requirement: " + std::to_string(measurements[1]));
        const auto& stats = solver.statistics;
        std::cout << mode << ',' << scene << ',' << config.resolution.x << ',' << config.resolution.y << ',' << config.resolution.z << ',' << sample << ',' << solver.time << ',' << wall_ms << ',' << gpu_ms << ',' << iterations << ',' << state.max_residual << ',' << measurements[1] << ',' << measurements[0] << ',' << solver.pressure.view.level_count << ',' << solver.pool.attribute(::cuda::memory_pool_attributes::used_mem_current) << ',' << stats.geometry_milliseconds << ',' << stats.advection_milliseconds << ',' << stats.forces_milliseconds << ',' << stats.projection_milliseconds << ',' << stats.statistics_milliseconds << ',' << stats.substeps << '\n';
    }
    if (audit) {
        auto download = [&]<typename T>(T* pointer) {
            std::vector<T> data(cells);
            ::cuda::copy_bytes(solver.stream, ::cuda::std::span(pointer, cells), ::cuda::std::span(data.data(), cells), {.src_access_order = ::cuda::source_access_order::stream});
            solver.synchronize();
            return data;
        };
        auto fluid            = download(solver.grid.fluid);
        auto anchors          = download(solver.pressure.view.anchors);
        auto potential        = download(solver.pressure.view.solution);
        auto rhs              = download(solver.pressure.hierarchy[0].rhs);
        auto boundary         = download(solver.pressure.hierarchy[0].boundary);
        auto solid_divergence = download(solver.grid.solid_divergence);
        std::array<std::vector<float>, 3> velocity, aperture, solid_velocity, face;
        for (int axis = 0; axis < 3; ++axis) {
            velocity[axis]       = download(solver.grid.state.velocity[axis]);
            aperture[axis]       = download(solver.grid.aperture[axis]);
            solid_velocity[axis] = download(solver.grid.solid_velocity[axis]);
            face[axis]           = download(solver.pressure.hierarchy[0].face[axis]);
        }
        auto layout = solver.grid.layout;
        std::array<int, 3> stride{1, layout.pitch, layout.slice};
        double max_divergence = 0, max_residual = 0, squared_residual = 0, squared_rhs = 0;
        int anchor_count = 0;
        for (int z = 0; z < layout.size.z; ++z)
            for (int y = 0; y < layout.size.y; ++y)
                for (int x = 0; x < layout.size.x; ++x) {
                    int i = x + 1 + (y + 1) * layout.pitch + (z + 1) * layout.slice;
                    if (!fluid[i]) continue;
                    double flux = 0, product = static_cast<double>(boundary[i]) * potential[i];
                    for (int axis = 0; axis < 3; ++axis) {
                        int left = i - stride[axis], right = i + stride[axis];
                        double a = solver.grid.boundary_mode == pyro::BoundaryMode::none ? 1 : aperture[axis][i];
                        double b = solver.grid.boundary_mode == pyro::BoundaryMode::none ? 1 : aperture[axis][right];
                        flux += b * velocity[axis][right] - a * velocity[axis][i];
                        if (solver.grid.boundary_mode == pyro::BoundaryMode::moving) flux += (1 - b) * solid_velocity[axis][right] - (1 - a) * solid_velocity[axis][i];
                        product += static_cast<double>(face[axis][i]) * (potential[i] - potential[left]);
                        product += static_cast<double>(face[axis][right]) * (potential[i] - potential[right]);
                    }
                    double divergence = flux / config.cell_size - (solver.grid.boundary_mode == pyro::BoundaryMode::moving ? solid_divergence[i] : 0);
                    double residual   = rhs[i] - product;
                    max_divergence    = std::max(max_divergence, std::isfinite(divergence) ? std::abs(divergence) : INFINITY);
                    max_residual      = std::max(max_residual, std::isfinite(residual) ? std::abs(residual) : INFINITY);
                    squared_residual += residual * residual;
                    squared_rhs += static_cast<double>(rhs[i]) * rhs[i];
                    anchor_count += anchors[i] != 0;
                }
        std::cerr << std::setprecision(12) << "{\"cpu_divergence\":" << max_divergence << ",\"cpu_true_residual\":" << max_residual << ",\"relative_l2_residual\":" << std::sqrt(squared_residual / squared_rhs) << ",\"anchors\":" << anchor_count << "}\n";
        if (!std::isfinite(max_divergence) || max_divergence > config.pressure.divergence_tolerance || max_residual > config.pressure.divergence_tolerance * .5) throw std::runtime_error("CPU projection audit failed");
    }
    if (!snapshot.empty()) {
        std::ofstream output(snapshot, std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        auto layout = solver.grid.layout;
        std::array<int, 6> header{layout.size.x, layout.size.y, layout.size.z, layout.pitch, layout.slice, layout.cells};
        output.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
        std::vector<float> host(cells);
        for (auto* field : {solver.grid.state.velocity[0], solver.grid.state.velocity[1], solver.grid.state.velocity[2], solver.grid.state.smoke, solver.grid.state.temperature}) {
            ::cuda::copy_bytes(solver.stream, ::cuda::std::span(field, cells), ::cuda::std::span(host.data(), cells), {.src_access_order = ::cuda::source_access_order::stream});
            solver.synchronize();
            output.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(float));
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
