module;
#include "cuda.cuh"
export module pyro.dense;
export import pyro.types;
import std;
export namespace pyro::dense {
    struct Config {
        float cellSize{0.05f};
        std::array<float, 3> origin{-3.2f, -0.8f, -3.2f};
        std::array<int, 3> resolution{128, 192, 128};
        float cfl{0.75f};
        float maxTimeStep{1.0f / 48.0f};
        float buoyancy{1.0f};
        float smokeWeight{0.05f};
        std::array<float, 3> up{0.0f, 1.0f, 0.0f};
        std::array<float, 3> wind{};
        float cooling{0.1f};
        float dissipation{0.01f};
        float pressureRelativeTolerance{1.0e-5f};
        float divergenceTolerance{2.0e-4f}; // Absolute max divergence, in inverse seconds.
        int pressureMaxIterations{200};
    };

    struct Statistics {
        int fluidCells{}, substeps{}, pressureIterations{}, pressureLevels{};
        float maxDivergence{}, maxSpeed{}, stepMilliseconds{};
        float geometryMilliseconds{}, advectionMilliseconds{}, forcesMilliseconds{}, projectionMilliseconds{}, statisticsMilliseconds{};
        double smokeIntegral{}, temperatureIntegral{};
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
        void set_sources(const Source* data, int count);
        void set_colliders(const Collider* data, int count);
        int add_mesh(std::span<const Triangle> triangles);
        void advance(double duration);
        void synchronize();
        // Restore into the same domain/configuration; scene inputs remain caller-owned.
        void checkpoint(const char* path, bool restore);
        std::span<const float> download_density();

    private:
        const cuda::dense::Config parameters;
        cudaStream_t stream{};
        cuda::dense::Grid grid;
        cuda::dense::PressureWorkspace pressure;
        cuda::Source* sources{};
        cuda::Collider* colliders{};
        int sourceCount{}, colliderCount{}, sourceCapacity{}, colliderCapacity{};
        std::vector<cuda::Mesh> meshes;
        std::vector<cuda::dense::SourceRegion> sourceRegions;
        std::vector<Source> sourceInputs;
        std::vector<Collider> colliderInputs;
        bool geometryDirty{true}, matrixDirty{true};
        float* maximum{};
        std::vector<float> density;
        cudaEvent_t events[10]{};

        template <typename T>
        void allocate(T*& pointer, unsigned long long count);
        template <typename T>
        void release(T*& pointer, unsigned long long count);
        void project(cuda::dense::Fields fields);
        void release_storage();
    };
} // namespace pyro::dense
