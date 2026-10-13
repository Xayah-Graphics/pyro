module;
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#endif

module pyro.stream;
import std;
import pyro.sparse;

namespace pyro {
    void run_stream() {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        std::cin.exceptions(std::ios::badbit | std::ios::failbit);
        std::cout.exceptions(std::ios::badbit | std::ios::failbit);
        sparse::Config config;
        static_assert(sizeof(sparse::Config) == 92);
        static_assert(sizeof(Motion) == 128);
        static_assert(sizeof(Triangle) == 36);
        std::cin.read(reinterpret_cast<char*>(&config), sizeof(config));
        sparse::Solver solver(config);
        int meshes;
        std::cin.read(reinterpret_cast<char*>(&meshes), sizeof(meshes));
        for (int i = 0; i < meshes; ++i) {
            int count;
            std::cin.read(reinterpret_cast<char*>(&count), sizeof(count));
            std::vector<Triangle> triangles(count);
            std::cin.read(reinterpret_cast<char*>(triangles.data()), count * sizeof(Triangle));
            solver.add_mesh(triangles);
        }
        auto read  = []<typename T>(T& value) { std::cin.read(reinterpret_cast<char*>(&value), sizeof(T)); };
        auto write = []<typename T>(const T& value) { std::cout.write(reinterpret_cast<const char*>(&value), sizeof(T)); };
        std::vector<Source> sources;
        std::vector<Collider> colliders;
        float frameMilliseconds = 0;
        int frameSubsteps = 0, frameIterations = 0;
        write('K');
        std::cout.flush();
        for (;;) {
            char command;
            read(command);
            if (command == 'Q') break;
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
                read(frame);
                auto begin               = std::chrono::steady_clock::now();
                auto density             = solver.download_density();
                float exportMilliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
                begin                    = std::chrono::steady_clock::now();
                auto grid                = solver.download_grid();
                float gridMilliseconds   = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - begin).count();
                const auto& stats        = solver.statistics;
                // <iidfffiifQfiidif, density records, then all allocated grid records.
                static_assert(sizeof(sparse::DensityBlock) == 2060);
                static_assert(sizeof(sparse::GridBlock) == 20);
                write(frame);
                write(static_cast<int>(density.size()));
                write(solver.time);
                write(config.cellSize);
                write(frameMilliseconds);
                write(exportMilliseconds);
                write(stats.blocks);
                write(stats.fluidCells);
                write(stats.maxDivergence);
                write(stats.allocatedBytes);
                write(stats.maxSpeed);
                write(frameSubsteps);
                write(frameIterations);
                write(stats.smokeIntegral);
                write(config.maxBlocks);
                write(gridMilliseconds);
                if (!density.empty()) std::cout.write(reinterpret_cast<const char*>(density.data()), static_cast<std::streamsize>(density.size_bytes()));
                if (!grid.empty()) std::cout.write(reinterpret_cast<const char*>(grid.data()), static_cast<std::streamsize>(grid.size_bytes()));
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
    }
} // namespace pyro
