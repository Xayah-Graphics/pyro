module;
#include "cuda.cuh"
export module pyro.sparse;
export import pyro.types;
import std;

export namespace pyro::sparse {
    struct Config {
        float cellSize{0.05f}; // Finest spacing; level l has spacing cellSize * 2^l.
        int initialBlocks{4096}; // Initial reservation; storage grows with the leaf count.
        int levels{3}, pressurePadding{1}, adaptInterval{2}, coarsenDelay{4}, keepAlive{4};
        float smokeAbsoluteError{0.0003125f}, temperatureAbsoluteError{0.00125f}, velocityAbsoluteError{0.001875f}, relativeError{0.003125f};
        float smokeThreshold{1.e-5f}, temperatureThreshold{1.e-4f}, velocityThreshold{0.025f};
        float cfl{0.75f}, maxTimeStep{1.f / 48}, buoyancy{1}, smokeWeight{0.05f};
        std::array<float, 3> up{0, 1, 0}, wind{};
        float cooling{0.1f}, dissipation{0.01f}, pressureRelativeTolerance{1.e-5f}, divergenceTolerance{2.e-4f};
        int pressureMaxIterations{300};
    };
    struct GridBlock {
        std::array<int, 3> coordinate{}; // In blocks at this level, not finest cells.
        int level{}, quiet{}, remaining{};
    };
    struct DensityBlock {
        GridBlock block;
        std::array<float, 512> values;
    };
    struct Statistics {
        int blocks{}, capacity{}, fluidCells{}, substeps{}, pressureIterations{}, pressureLevels{};
        float maxDivergence{}, maxSpeed{}, stepMilliseconds{}, topologyMilliseconds{}, geometryMilliseconds{}, advectionMilliseconds{}, forcesMilliseconds{}, projectionMilliseconds{}, statisticsMilliseconds{};
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
        void checkpoint(const char* path, bool restore);
        std::span<const DensityBlock> download_density();

    private:
        const cuda::sparse::Config parameters;
        cudaStream_t stream{};
        cuda::sparse::Grid grid, spare;
        cuda::sparse::Topology topology;
        cuda::sparse::Scratch scratch;
        cuda::sparse::PressureWorkspace pressure;
        cuda::Source* sources{};
        cuda::Collider* colliders{};
        int sourceCount{}, colliderCount{}, sourceCapacity{}, colliderCapacity{}, substep{};
        std::vector<cuda::Mesh> meshes;
        std::vector<Source> sourceInputs;
        std::vector<Collider> colliderInputs;
        bool topologyDirty{true}, geometryDirty{true}, movingGeometry{}, matrixDirty{true};
        std::vector<DensityBlock> density;
        std::vector<std::array<cudaEvent_t, 2>> events;

        template <typename T>
        void allocate(T*& pointer, unsigned long long count, unsigned long long zeroFrom = 0);
        template <typename T>
        void release(T*& pointer, unsigned long long count);
        template <typename T>
        void resize(T*& pointer, unsigned long long previousCount, unsigned long long count, unsigned long long retained = 0);
        void reserve(int required);
        void build_pressure();
        void project(cuda::sparse::Fields fields);
        void release_storage();
    };
} // namespace pyro::sparse
