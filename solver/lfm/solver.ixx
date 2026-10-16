module;
#include "cuda.cuh"
export module pyro.lfm;
export import pyro.types;
import std;

export namespace pyro::lfm {
    struct Config {
        float cell_size{1.0f / 128};
        std::array<float, 3> origin{};
        std::array<int, 3> resolution{256, 128, 128};
        std::array<float, 3> wind{0.5638156f, 0.2052121f, 0};
        int cycle_steps{5};
        float pressure_relative_tolerance{1.e-5f}, divergence_tolerance{2.e-4f};
        int pressure_max_iterations{200};
    };
    struct Statistics {
        int fluid_cells{}, substeps{}, pressure_iterations{}, pressure_levels{};
        float max_divergence{}, max_speed{}, step_milliseconds{};
        float geometry_milliseconds{}, advection_milliseconds{}, flow_map_milliseconds{}, projection_milliseconds{}, statistics_milliseconds{};
        unsigned long long allocated_bytes{};
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
        int collider_count{};
        bool geometry_dirty{true};

        template <typename T>
        void allocate(T*& pointer, std::size_t count);
        void prepare();
        void project(cuda::lfm::Velocity velocity);
        void release_storage();
    };
} // namespace pyro::lfm
