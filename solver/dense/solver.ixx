module;
#include "cuda.cuh"
export module pyro.dense;
export import pyro.types;
import std;
export namespace pyro::dense {
    struct Config {
        float cell_size{0.05f};
        std::array<float, 3> origin{-3.2f, -0.8f, -3.2f};
        std::array<int, 3> resolution{128, 192, 128};
        float cfl{0.75f};
        float max_time_step{1.0f / 48.0f};
        float buoyancy{1.0f};
        float smoke_weight{0.05f};
        std::array<float, 3> up{0.0f, 1.0f, 0.0f};
        std::array<float, 3> wind{};
        float cooling{0.1f};
        float dissipation{0.01f};
        float pressure_relative_tolerance{1.0e-5f};
        float divergence_tolerance{2.0e-4f};
        int pressure_max_iterations{200};
    };

    struct Statistics {
        int fluid_cells{}, substeps{}, pressure_iterations{}, pressure_levels{};
        float max_divergence{}, max_speed{}, step_milliseconds{};
        float geometry_milliseconds{}, advection_milliseconds{}, forces_milliseconds{}, projection_milliseconds{}, statistics_milliseconds{};
        double smoke_integral{}, temperature_integral{};
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
        void set_sources(const Source* data, int count);
        void set_colliders(const Collider* data, int count);
        int add_mesh(std::span<const Triangle> triangles);
        void advance(double duration);
        void synchronize();
        void checkpoint(const char* path, bool restore);
        std::span<const float> download_density();

    private:
        const cuda::dense::Config parameters;
        cudaStream_t stream{};
        cuda::dense::Grid grid;
        cuda::dense::PressureWorkspace pressure;
        cuda::Source* sources{};
        cuda::Collider* colliders{};
        int source_count{}, collider_count{}, source_capacity{}, collider_capacity{};
        std::vector<cuda::Mesh> meshes;
        std::vector<cuda::dense::SourceRegion> source_regions;
        std::vector<Source> source_inputs;
        std::vector<Collider> collider_inputs;
        bool geometry_dirty{true}, matrix_dirty{true};
        float* maximum{};
        std::vector<float> density;
        cudaEvent_t events[9]{};

        template <typename T>
        void allocate(T*& pointer, unsigned long long count, bool clear = true);
        template <typename T>
        void release(T*& pointer, unsigned long long count);
        void project(cuda::dense::Fields fields);
        void release_storage();
    };
} // namespace pyro::dense
