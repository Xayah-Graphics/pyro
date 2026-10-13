module;

#include "cuda.cuh"

export module pyro.sparse;
export import pyro.types;

import std;

export namespace pyro::sparse {
    struct Config {
        float cellSize{0.05f};
        int maxBlocks{4096};
        int pressurePadding{2}; // In 8-cell blocks; independent of the advection stencil.
        int keepAliveSteps{4};
        float smokeThreshold{1.0e-5f};
        float temperatureThreshold{1.0e-4f};
        float velocityThreshold{0.025f};
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
        int maxPressureLevels{12};
    };

    struct DensityBlock {
        std::array<int, 3> coordinate{};
        float values[512]; // x contiguous; cell-centered samples.
    };

    // Low bits are storage/domain/new allocation; higher bits explain the last topology request.
    inline constexpr unsigned gridSmoke = 8u, gridTemperature = 16u, gridVelocity = 32u;
    inline constexpr unsigned gridEmitter = 64u, gridCollider = 128u, gridRetained = 256u, gridPadding = 512u;

    struct GridBlock {
        std::array<int, 3> coordinate{};
        unsigned flags;
        int remaining; // Future inactive substeps that will still retain this block as a seed.
    };

    struct Statistics {
        int blocks{}, fluidCells{}, substeps{}, pressureIterations{}, pressureLevels{};
        float maxDivergence{}, maxSpeed{}, stepMilliseconds{}, topologyMilliseconds{}, projectionMilliseconds{};
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
        // Frame-boundary state, including sparse allocation history. Workspaces are rebuilt.
        void checkpoint(const char* path, bool restore);
        // Borrowed pinned storage, valid until the next download or destruction.
        std::span<const DensityBlock> download_density();
        std::span<const GridBlock> download_grid();

    private:
        const cuda::Config parameters;
        cudaStream_t stream{};
        cuda::Grid grid;
        cuda::PressureWorkspace pressure;
        cuda::Source* sources{};
        cuda::Collider* colliders{};
        int sourceCount{}, colliderCount{};
        int sourceCapacity{}, colliderCapacity{};
        std::vector<cuda::Mesh> meshes;
        std::vector<Collider> colliderInputs;
        int* error{};
        float* maximum{};
        bool hierarchyDirty{true};
        bool geometryDirty{true};
        cuda::DensityBlock* densityDevice{};
        DensityBlock* densityHost{};
        int* densityCount{};
        cuda::GridBlock* gridDevice{};
        GridBlock* gridHost{};

        template <typename T>
        void allocate(T*& pointer, unsigned long long count);
        template <typename T>
        void release(T*& pointer, unsigned long long count);
        void allocate_layout(cuda::Layout& layout, int capacity);
        void free_layout(cuda::Layout& layout);
        void project(cuda::Fields fields);
    };
} // namespace pyro::sparse
