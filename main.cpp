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
    pyro::sparse::Config sparseConfig;
    pyro::dense::Config denseConfig;
    pyro::lfm::Config lfmConfig;
    std::string backend = "sparse", checkpoint, resume;
    int frames          = 24;
    for (int i = 1; i < argc; ++i) {
        std::string_view option = argv[i];
        if (option == "--solver") backend = argv[++i];
        else if (option == "--frames") frames = std::stoi(argv[++i]);
        else if (option == "--cell-size") sparseConfig.cellSize = denseConfig.cellSize = lfmConfig.cellSize = std::stof(argv[++i]);
        else if (option == "--origin")
            for (int axis = 0; axis < 3; ++axis) denseConfig.origin[axis] = lfmConfig.origin[axis] = std::stof(argv[++i]);
        else if (option == "--resolution")
            for (int axis = 0; axis < 3; ++axis) denseConfig.resolution[axis] = lfmConfig.resolution[axis] = std::stoi(argv[++i]);
        else if (option == "--wind")
            for (int axis = 0; axis < 3; ++axis) sparseConfig.wind[axis] = denseConfig.wind[axis] = lfmConfig.wind[axis] = std::stof(argv[++i]);
        else if (option == "--cycle-steps") lfmConfig.cycleSteps = std::stoi(argv[++i]);
        else if (option == "--initial-blocks") sparseConfig.initialBlocks = std::stoi(argv[++i]);
        else if (option == "--levels") sparseConfig.levels = std::stoi(argv[++i]);
        else if (option == "--adapt-interval") sparseConfig.adaptInterval = std::stoi(argv[++i]);
        else if (option == "--relative-error") sparseConfig.relativeError = std::stof(argv[++i]);
        else if (option == "--smoke-error") sparseConfig.smokeAbsoluteError = std::stof(argv[++i]);
        else if (option == "--temperature-error") sparseConfig.temperatureAbsoluteError = std::stof(argv[++i]);
        else if (option == "--velocity-error") sparseConfig.velocityAbsoluteError = std::stof(argv[++i]);
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
            source.temperatureRate = 3.0f;
            solver.set_sources(&source, 1);
        }
        if (!resume.empty()) solver.checkpoint(resume.c_str(), true);
        for (int frame = 0; frame < frames; ++frame) {
            solver.advance(backend == "lfm" ? 1.0 / 20 : 1.0 / 24);
            const auto& stats = solver.statistics;
            std::cout << "solver=" << backend << " frame=" << frame + 1 << " time=" << solver.time << " fluid=" << stats.fluidCells << " substeps=" << stats.substeps << " divergence=" << stats.maxDivergence << " step=" << stats.stepMilliseconds << "ms advection=" << stats.advectionMilliseconds << "ms pressure=" << stats.projectionMilliseconds << "ms iterations=" << stats.pressureIterations << " memory=" << stats.allocatedBytes / 1048576.0 << "MiB";
            if constexpr (std::same_as<std::remove_cvref_t<decltype(solver)>, pyro::lfm::Solver>) std::cout << " maps=" << stats.flowMapMilliseconds << "ms";
            else std::cout << " smoke=" << stats.smokeIntegral;
            std::cout << '\n';
        }
        if (!checkpoint.empty()) solver.checkpoint(checkpoint.c_str(), false);
    };
    if (backend == "lfm") {
        pyro::lfm::Solver solver(lfmConfig);
        run(solver);
    } else if (backend == "dense") {
        pyro::dense::Solver solver(denseConfig);
        run(solver);
    } else if (backend == "sparse") {
        pyro::sparse::Solver solver(sparseConfig);
        run(solver);
    } else throw std::runtime_error("Unknown solver: " + backend);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
