import std;
import pyro.sparse;
import pyro.dense;
import pyro.lfm;
import pyro.stream;

int main(int argc, char** argv) try {
    if (argc > 1 && std::string_view(argv[1]) == "--stream") {
        pyro::run_stream();
        return 0;
    }
    pyro::sparse::Config sparse_config;
    pyro::dense::Config dense_config;
    pyro::lfm::Config lfm_config;
    std::string backend = "sparse", checkpoint, resume;
    int frames          = 24;
    for (int i = 1; i < argc; ++i) {
        std::string_view option = argv[i];
        if (option == "--solver") backend = argv[++i];
        else if (option == "--frames") frames = std::stoi(argv[++i]);
        else if (option == "--cell-size") sparse_config.cell_size = dense_config.cell_size = lfm_config.cell_size = std::stof(argv[++i]);
        else if (option == "--origin")
            for (int axis = 0; axis < 3; ++axis) dense_config.origin[axis] = lfm_config.origin[axis] = std::stof(argv[++i]);
        else if (option == "--resolution")
            for (int axis = 0; axis < 3; ++axis) dense_config.resolution[axis] = lfm_config.resolution[axis] = std::stoi(argv[++i]);
        else if (option == "--wind")
            for (int axis = 0; axis < 3; ++axis) sparse_config.wind[axis] = dense_config.wind[axis] = lfm_config.wind[axis] = std::stof(argv[++i]);
        else if (option == "--cycle-steps") lfm_config.cycle_steps = std::stoi(argv[++i]);
        else if (option == "--initial-blocks") sparse_config.initial_blocks = std::stoi(argv[++i]);
        else if (option == "--levels") sparse_config.levels = std::stoi(argv[++i]);
        else if (option == "--adapt-interval") sparse_config.adapt_interval = std::stoi(argv[++i]);
        else if (option == "--relative-error") sparse_config.relative_error = std::stof(argv[++i]);
        else if (option == "--smoke-error") sparse_config.smoke_absolute_error = std::stof(argv[++i]);
        else if (option == "--temperature-error") sparse_config.temperature_absolute_error = std::stof(argv[++i]);
        else if (option == "--velocity-error") sparse_config.velocity_absolute_error = std::stof(argv[++i]);
        else if (option == "--checkpoint") checkpoint = argv[++i];
        else if (option == "--resume") resume = argv[++i];
        else if (option == "--help") {
            std::cout << "pyro --solver sparse|dense|lfm --frames N --cell-size H\n"
                         "     --origin X Y Z --resolution NX NY NZ\n"
                         "     --initial-blocks N --levels N --adapt-interval N --relative-error E\n"
                         "     --smoke-error E --temperature-error E --velocity-error E\n"
                         "     --checkpoint PATH --resume PATH\n";
            std::cout << "     --wind X Y Z --cycle-steps N (LFM: one cycle per 1/20 s)\n";
            return 0;
        } else throw std::runtime_error("Unknown option: " + std::string(option));
    }
    auto run = [&](auto& solver) {
        if constexpr (!std::same_as<std::remove_cvref_t<decltype(solver)>, pyro::lfm::Solver>) {
            pyro::Source source;
            source.center          = {0.0f, 0.25f, 0.0f};
            source.velocity        = {0.0f, 1.0f, 0.0f};
            source.temperature_rate = 3.0f;
            solver.set_sources(&source, 1);
        }
        if (!resume.empty()) solver.checkpoint(resume.c_str(), true);
        for (int frame = 0; frame < frames; ++frame) {
            solver.advance(backend == "lfm" ? 1.0 / 20 : 1.0 / 24);
            const auto& stats = solver.statistics;
            std::cout << "solver=" << backend << " frame=" << frame + 1 << " time=" << solver.time << " fluid=" << stats.fluid_cells << " substeps=" << stats.substeps << " divergence=" << stats.max_divergence << " step=" << stats.step_milliseconds << "ms advection=" << stats.advection_milliseconds << "ms pressure=" << stats.projection_milliseconds << "ms iterations=" << stats.pressure_iterations << " memory=" << stats.allocated_bytes / 1048576.0 << "MiB";
            if constexpr (std::same_as<std::remove_cvref_t<decltype(solver)>, pyro::lfm::Solver>) std::cout << " maps=" << stats.flow_map_milliseconds << "ms";
            else std::cout << " smoke=" << stats.smoke_integral;
            std::cout << '\n';
        }
        if (!checkpoint.empty()) solver.checkpoint(checkpoint.c_str(), false);
    };
    if (backend == "lfm") {
        pyro::lfm::Solver solver(lfm_config);
        run(solver);
    } else if (backend == "dense") {
        pyro::dense::Solver solver(dense_config);
        run(solver);
    } else if (backend == "sparse") {
        pyro::sparse::Solver solver(sparse_config);
        run(solver);
    } else throw std::runtime_error("Unknown solver: " + backend);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
