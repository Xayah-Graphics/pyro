#include <array>
#include <chrono>
#include <concepts>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <solver/dense/solver.cuh>
#include <solver/lfm/solver.cuh>
#include <solver/sparse/solver.cuh>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#endif

int main() try {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    std::cin.exceptions(std::ios::badbit | std::ios::failbit);
    std::cout.exceptions(std::ios::badbit | std::ios::failbit);
    auto read  = []<typename T>(T& value) { std::cin.read(reinterpret_cast<char*>(&value), sizeof(T)); };
    auto write = []<typename T>(const T& value) { std::cout.write(reinterpret_cast<const char*>(&value), sizeof(T)); };
    auto run   = [&]<typename Solver>(const auto& config) {
        Solver solver(config);
        float cell_size;
        if constexpr (std::same_as<Solver, pyro::sparse::Solver>) cell_size = config.grid.cell_size;
        else cell_size = config.cell_size;
        int meshes;
        read(meshes);
        for (int i = 0; i < meshes; ++i) {
            int point_count, triangle_count;
            read(point_count);
            read(triangle_count);
            std::vector<float3> positions(point_count);
            std::vector<int3> triangles(triangle_count);
            std::cin.read(reinterpret_cast<char*>(positions.data()), positions.size() * sizeof(positions[0]));
            std::cin.read(reinterpret_cast<char*>(triangles.data()), triangles.size() * sizeof(triangles[0]));
            solver.scene.add_mesh(positions, triangles);
        }
        int instance_count;
        read(instance_count);
        solver.scene.instances.resize(instance_count);
        for (auto& instance : solver.scene.instances) {
            read(instance.shape);
            instance.motion = {.kind = geometry::MotionKind::affine, .affine = {}};
        }
        if constexpr (std::same_as<Solver, pyro::lfm::Solver>) {
            int updates, count;
            read(updates);
            read(count);
            for (int i = 0; i < updates; ++i) {
                int instance;
                read(instance);
                read(solver.scene.instances[instance].motion.affine.matrix);
                read(solver.scene.instances[instance].motion.affine.derivative);
                read(solver.scene.instances[instance].motion.affine.time);
            }
            solver.colliders.resize(count);
            for (auto& collider : solver.colliders) {
                read(collider.instance);
                read(collider.surface_offset);
            }
        }
        float frame_milliseconds = 0;
        int frame_substeps = 0, frame_iterations = 0;
        write('K');
        std::cout.flush();
        for (;;) {
            char command;
            read(command);
            if (command == 'A') {
                double end_time;
                int updates, source_count, collider_count;
                read(end_time);
                if constexpr (!std::same_as<Solver, pyro::lfm::Solver>) {
                    read(updates);
                    read(source_count);
                    read(collider_count);
                    for (int i = 0; i < updates; ++i) {
                        int instance;
                        read(instance);
                        read(solver.scene.instances[instance].motion.affine.matrix);
                        read(solver.scene.instances[instance].motion.affine.derivative);
                        read(solver.scene.instances[instance].motion.affine.time);
                    }
                    solver.sources.resize(source_count);
                    solver.colliders.resize(collider_count);
                    for (auto& source : solver.sources) {
                        read(source.instance);
                        read(source.smoke_rate);
                        read(source.temperature_rate);
                        read(source.velocity_rate);
                        read(source.velocity);
                        read(source.inherit_velocity);
                    }
                    for (auto& collider : solver.colliders) {
                        read(collider.instance);
                        read(collider.surface_offset);
                    }
                }
                solver.advance(end_time - solver.time);
                frame_milliseconds += solver.statistics.wall_milliseconds;
                frame_substeps += solver.statistics.substeps;
                frame_iterations += solver.statistics.pressure_iterations;
                write('K');
            } else if (command == 'F') {
                int frame;
                bool snapshot;
                read(frame);
                read(snapshot);
                if (snapshot) {
                    auto begin   = std::chrono::steady_clock::now();
                    auto density = [&] {
                        if constexpr (std::same_as<Solver, pyro::lfm::Solver>) return solver.download_vorticity();
                        else return solver.download_density();
                    }();
                    float export_milliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
                    int blocks = 0, capacity = 0, levels = 1;
                    float3 origin{};
                    int3 resolution{};
                    if constexpr (std::same_as<Solver, pyro::sparse::Solver>) {
                        blocks   = solver.statistics.blocks;
                        capacity = solver.statistics.capacity;
                        levels   = config.grid.levels;
                    } else {
                        origin     = config.origin;
                        resolution = config.resolution;
                    }
                    const auto& stats = solver.statistics;
                    // <iidfffiifQfiidi3f4i. Sparse density records include AMR leaf metadata.
                    static_assert(sizeof(pyro::sparse::DensityBlock) == 2072);
                    write(frame);
                    write(static_cast<int>(density.size()));
                    write(solver.time);
                    write(cell_size);
                    write(frame_milliseconds);
                    write(export_milliseconds);
                    write(blocks);
                    write(stats.fluid_cells);
                    write(stats.max_divergence);
                    write(stats.allocated_bytes);
                    write(stats.max_speed);
                    write(frame_substeps);
                    write(frame_iterations);
                    if constexpr (std::same_as<Solver, pyro::lfm::Solver>) write(0.0);
                    else write(stats.smoke_integral);
                    write(capacity);
                    write(origin);
                    write(resolution);
                    write(levels);
                    if (!density.empty()) std::cout.write(reinterpret_cast<const char*>(density.data()), static_cast<std::streamsize>(density.size_bytes()));
                } else write('K');
                frame_milliseconds = 0;
                frame_substeps = frame_iterations = 0;
            } else if (command == 'S' || command == 'L') {
                unsigned size;
                read(size);
                std::string path(size, '\0');
                std::cin.read(path.data(), size);
                solver.checkpoint(path.c_str(), command == 'L');
                write('K');
                if (command == 'L') write(solver.time);
            } else throw std::runtime_error("Unknown solver command");
            std::cout.flush();
        }
    };
    std::array<char, 8> protocol;
    read(protocol);
    if (protocol != std::array{'P', 'Y', 'R', 'O', 'S', 'T', 'R', '5'}) throw std::runtime_error("Unsupported solver stream format");
    char algorithm;
    read(algorithm);
    if (algorithm == 'S') {
        pyro::sparse::Config config;
        read(config.grid.cell_size);
        read(config.initial_blocks);
        read(config.grid.levels);
        read(config.grid.pressure_padding);
        read(config.grid.adapt_interval);
        read(config.grid.coarsen_delay);
        read(config.grid.keep_alive);
        read(config.grid.smoke_absolute_error);
        read(config.grid.temperature_absolute_error);
        read(config.grid.velocity_absolute_error);
        read(config.grid.relative_error);
        read(config.grid.smoke_threshold);
        read(config.grid.temperature_threshold);
        read(config.grid.velocity_threshold);
        read(config.grid.cfl);
        read(config.max_time_step);
        read(config.forces.buoyancy);
        read(config.forces.smoke_weight);
        read(config.forces.up);
        read(config.grid.wind);
        read(config.forces.cooling);
        read(config.forces.dissipation);
        read(config.pressure.relative_tolerance);
        read(config.pressure.divergence_tolerance);
        read(config.pressure.max_iterations);
        run.operator()<pyro::sparse::Solver>(config);
    } else if (algorithm == 'D') {
        pyro::dense::Config config;
        read(config.cell_size);
        read(config.origin);
        read(config.resolution);
        read(config.cfl);
        read(config.max_time_step);
        read(config.forces.buoyancy);
        read(config.forces.smoke_weight);
        read(config.forces.up);
        read(config.wind);
        read(config.forces.cooling);
        read(config.forces.dissipation);
        read(config.pressure.relative_tolerance);
        read(config.pressure.divergence_tolerance);
        read(config.pressure.max_iterations);
        run.operator()<pyro::dense::Solver>(config);
    } else if (algorithm == 'L') {
        pyro::lfm::Config config;
        read(config.cell_size);
        read(config.origin);
        read(config.resolution);
        read(config.wind);
        read(config.cycle_steps);
        read(config.pressure.relative_tolerance);
        read(config.pressure.divergence_tolerance);
        read(config.pressure.max_iterations);
        run.operator()<pyro::lfm::Solver>(config);
    } else {
        throw std::runtime_error("Unknown solver algorithm");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
