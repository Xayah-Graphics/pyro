import std;
import pyro.sparse;
import pyro.dense;
import pyro.stream;

int main(int argc, char** argv) try {
    if (argc > 1 && std::string_view(argv[1]) == "--stream") {
        pyro::run_stream();
        return 0;
    }
    pyro::sparse::Config sparseConfig;
    pyro::dense::Config denseConfig;
    std::string backend = "sparse", checkpoint, resume;
    int frames          = 24;
    for (int i = 1; i < argc; ++i) {
        std::string_view option = argv[i];
        if (option == "--solver") backend = argv[++i];
        else if (option == "--frames") frames = std::stoi(argv[++i]);
        else if (option == "--cell-size") sparseConfig.cellSize = denseConfig.cellSize = std::stof(argv[++i]);
        else if (option == "--origin")
            for (auto& value : denseConfig.origin) value = std::stof(argv[++i]);
        else if (option == "--resolution")
            for (auto& value : denseConfig.resolution) value = std::stoi(argv[++i]);
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
            std::cout << "pyro --solver sparse|dense --frames N --cell-size H\n"
                         "     --origin X Y Z --resolution NX NY NZ\n"
                         "     --initial-blocks N --levels N --adapt-interval N --relative-error E\n"
                         "     --smoke-error E --temperature-error E --velocity-error E\n"
                         "     --checkpoint PATH --resume PATH\n";
            return 0;
        } else throw std::runtime_error("Unknown option: " + std::string(option));
    }
    auto run = [&](auto& solver) {
        pyro::Source source;
        source.center          = {0.0f, 0.25f, 0.0f};
        source.velocity        = {0.0f, 1.0f, 0.0f};
        source.temperatureRate = 3.0f;
        solver.set_sources(&source, 1);
        if (!resume.empty()) solver.checkpoint(resume.c_str(), true);
        for (int frame = 0; frame < frames; ++frame) {
            solver.advance(1.0f / 24.0f);
            const auto& stats = solver.statistics;
            std::cout << "solver=" << backend << " frame=" << frame + 1 << " time=" << solver.time << " fluid=" << stats.fluidCells << " substeps=" << stats.substeps << " divergence=" << stats.maxDivergence << " smoke=" << stats.smokeIntegral << " step=" << stats.stepMilliseconds << "ms pressure=" << stats.projectionMilliseconds << "ms iterations=" << stats.pressureIterations << " memory=" << stats.allocatedBytes / 1048576.0 << "MiB\n";
        }
        if (!checkpoint.empty()) solver.checkpoint(checkpoint.c_str(), false);
    };
    if (backend == "dense") {
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
