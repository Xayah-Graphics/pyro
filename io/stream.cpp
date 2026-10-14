module;
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#endif

module pyro.stream;
import std;
import pyro.sparse;
import pyro.dense;

namespace pyro {
    void run_stream() {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        std::cin.exceptions(std::ios::badbit | std::ios::failbit);
        std::cout.exceptions(std::ios::badbit | std::ios::failbit);
        static_assert(sizeof(sparse::Config) == 116);
        static_assert(sizeof(dense::Config) == 88);
        static_assert(sizeof(Motion) == 128);
        static_assert(sizeof(Triangle) == 36);
        auto read  = []<typename T>(T& value) { std::cin.read(reinterpret_cast<char*>(&value), sizeof(T)); };
        auto write = []<typename T>(const T& value) { std::cout.write(reinterpret_cast<const char*>(&value), sizeof(T)); };
        auto run   = [&]<typename Solver>(const auto& config) {
            Solver solver(config);
            int meshes;
            read(meshes);
            for (int i = 0; i < meshes; ++i) {
                int count;
                read(count);
                std::vector<Triangle> triangles(count);
                std::cin.read(reinterpret_cast<char*>(triangles.data()), count * sizeof(Triangle));
                solver.add_mesh(triangles);
            }
            std::vector<Source> sources;
            std::vector<Collider> colliders;
            float frameMilliseconds = 0;
            int frameSubsteps = 0, frameIterations = 0;
            write('K');
            std::cout.flush();
            for (;;) {
                char command;
                read(command);
                if (command == 'A') {
                    double endTime;
                    int sourceCount, colliderCount;
                    read(endTime);
                    read(sourceCount);
                    read(colliderCount);
                    sources.resize(sourceCount);
                    colliders.resize(colliderCount);
                    for (auto& source : sources) {
                        read(source.mesh);
                        source.shape = Shape::mesh;
                        read(source.motion);
                        read(source.smokeRate);
                        read(source.temperatureRate);
                        read(source.velocityRate);
                        read(source.velocity);
                        read(source.inheritVelocity);
                    }
                    for (auto& collider : colliders) {
                        read(collider.mesh);
                        collider.shape = Shape::mesh;
                        read(collider.motion);
                        read(collider.surfaceOffset);
                    }
                    solver.set_sources(sources.data(), sourceCount);
                    solver.set_colliders(colliders.data(), colliderCount);
                    solver.advance(endTime - solver.time);
                    frameMilliseconds += solver.statistics.stepMilliseconds;
                    frameSubsteps += solver.statistics.substeps;
                    frameIterations += solver.statistics.pressureIterations;
                    write('K');
                } else if (command == 'F') {
                    int frame;
                    bool snapshot;
                    read(frame);
                    read(snapshot);
                    if (snapshot) {
                        auto begin               = std::chrono::steady_clock::now();
                        auto density             = solver.download_density();
                        float exportMilliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
                        int blocks = 0, capacity = 0, levels = 1;
                        std::array<float, 3> origin{};
                        std::array<int, 3> resolution{};
                        if constexpr (std::same_as<Solver, sparse::Solver>) {
                            blocks   = solver.statistics.blocks;
                            capacity = solver.statistics.capacity;
                            levels   = config.levels;
                        } else {
                            origin     = config.origin;
                            resolution = config.resolution;
                        }
                        const auto& stats = solver.statistics;
                        // <iidfffiifQfiidi3f4i. Sparse density records include AMR leaf metadata.
                        static_assert(sizeof(sparse::DensityBlock) == 2072);
                        write(frame);
                        write(static_cast<int>(density.size()));
                        write(solver.time);
                        write(config.cellSize);
                        write(frameMilliseconds);
                        write(exportMilliseconds);
                        write(blocks);
                        write(stats.fluidCells);
                        write(stats.maxDivergence);
                        write(stats.allocatedBytes);
                        write(stats.maxSpeed);
                        write(frameSubsteps);
                        write(frameIterations);
                        write(stats.smokeIntegral);
                        write(capacity);
                        write(origin);
                        write(resolution);
                        write(levels);
                        if (!density.empty()) std::cout.write(reinterpret_cast<const char*>(density.data()), static_cast<std::streamsize>(density.size_bytes()));
                    } else write('K');
                    frameMilliseconds = 0;
                    frameSubsteps = frameIterations = 0;
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
        if (protocol != std::array{'P', 'Y', 'R', 'O', 'S', 'T', 'R', '2'}) throw std::runtime_error("Unsupported solver stream format");
        char algorithm;
        read(algorithm);
        if (algorithm == 'S') {
            sparse::Config config;
            read(config);
            run.operator()<sparse::Solver>(config);
        } else if (algorithm == 'D') {
            dense::Config config;
            read(config);
            run.operator()<dense::Solver>(config);
        } else {
            throw std::runtime_error("Unknown solver algorithm");
        }
    }
} // namespace pyro
