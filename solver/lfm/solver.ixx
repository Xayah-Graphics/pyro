module;
#include "cuda.cuh"
export module pyro.lfm;
export import pyro.types;
import std;

export namespace pyro::lfm {
    struct Config {
        float cellSize{1.0f / 128};
        std::array<float, 3> origin{};
        std::array<int, 3> resolution{256, 128, 128};
        std::array<float, 3> wind{0.5638156f, 0.2052121f, 0};
        int cycleSteps{5};
        float pressureRelativeTolerance{1.e-5f}, divergenceTolerance{2.e-4f};
        int pressureMaxIterations{200};
    };
    struct Statistics {
        int fluidCells{}, substeps{}, pressureIterations{}, pressureLevels{};
        float maxDivergence{}, maxSpeed{}, stepMilliseconds{};
        float geometryMilliseconds{}, advectionMilliseconds{}, flowMapMilliseconds{}, projectionMilliseconds{}, statisticsMilliseconds{};
        unsigned long long allocatedBytes{};
    };
    struct Solver {
        const Config config;
        Statistics statistics;
        double time{};

        explicit Solver(const Config& settings = {});
        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        ~Solver();
        int add_mesh(std::span<const Triangle> triangles);
        void set_colliders(std::span<const Collider> colliders);
        // One complete reinitialization cycle; all cycleSteps use duration / cycleSteps.
        void advance(double duration);
        void synchronize();
        void checkpoint(const char* path, bool restore);
        std::span<const float> download_vorticity();

    private:
        const cuda::lfm::Config parameters;
        cudaStream_t stream{};
        cuda::lfm::Grid grid;
        cuda::lfm::PressureWorkspace pressure;
        std::vector<cuda::lfm::Velocity> history;
        std::vector<cuda::Mesh> meshes;
        std::vector<void*> storage;
        std::vector<cudaEvent_t> events;
        std::vector<float> output;
        cuda::Collider* colliders{};
        int colliderCount{};
        bool geometryDirty{true};

        template <typename T>
        void allocate(T*& pointer, std::size_t count);
        void prepare();
        void project(cuda::lfm::Velocity velocity);
        void release_storage();
    };
} // namespace pyro::lfm
