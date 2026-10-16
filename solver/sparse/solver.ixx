module;
#include "cuda.cuh"
export module pyro.sparse;
export import pyro.types;
import std;

export namespace pyro::sparse {
    struct Config {
        float cell_size{0.05f};
        int initial_blocks{4096};
        int levels{3}, pressure_padding{1}, adapt_interval{2}, coarsen_delay{4}, keep_alive{4};
        float smoke_absolute_error{0.0003125f}, temperature_absolute_error{0.00125f}, velocity_absolute_error{0.001875f}, relative_error{0.003125f};
        float smoke_threshold{1.e-5f}, temperature_threshold{1.e-4f}, velocity_threshold{0.025f};
        float cfl{0.75f}, max_time_step{1.f / 48}, buoyancy{1}, smoke_weight{0.05f};
        std::array<float, 3> up{0, 1, 0}, wind{};
        float cooling{0.1f}, dissipation{0.01f}, pressure_relative_tolerance{1.e-5f}, divergence_tolerance{2.e-4f};
        int pressure_max_iterations{300};
    };
    struct GridBlock {
        std::array<int, 3> coordinate{};
        int level{}, quiet{}, remaining{};
    };
    struct DensityBlock {
        GridBlock block;
        std::array<float, 512> values;
    };
    struct Statistics {
        int blocks{}, capacity{}, fluid_cells{}, substeps{}, pressure_iterations{}, pressure_levels{};
        float max_divergence{}, max_speed{}, step_milliseconds{}, topology_milliseconds{}, geometry_milliseconds{}, advection_milliseconds{}, forces_milliseconds{}, projection_milliseconds{}, statistics_milliseconds{};
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
        int source_count{}, collider_count{}, source_capacity{}, collider_capacity{}, substep{};
        std::vector<cuda::Mesh> meshes;
        std::vector<Source> source_inputs;
        std::vector<Collider> collider_inputs;
        bool topology_dirty{true}, geometry_dirty{true}, moving_geometry{}, matrix_dirty{true};
        std::vector<DensityBlock> density;
        std::vector<std::array<cudaEvent_t, 2>> events;

        template <typename T>
        void allocate(T*& pointer, unsigned long long count, unsigned long long zero_from = 0);
        template <typename T>
        void release(T*& pointer, unsigned long long count);
        template <typename T>
        void resize(T*& pointer, unsigned long long previous_count, unsigned long long count, unsigned long long retained = 0);
        void reserve(int required);
        void build_pressure();
        void project(cuda::sparse::Fields fields);
        void release_storage();
    };
} // namespace pyro::sparse
