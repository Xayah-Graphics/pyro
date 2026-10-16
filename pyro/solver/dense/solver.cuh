#ifndef PYRO_SOLVER_DENSE_SOLVER_CUH
#define PYRO_SOLVER_DENSE_SOLVER_CUH

#include <array>
#include <span>
#include <vector>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#include "forces.cuh"
#include "grid.cuh"
#include "pressure.cuh"
#include "statistics.cuh"
#include <geometry/geometry.cuh>
#include <solver/common/scene.cuh>

namespace pyro::dense {
    struct Config {
        float cell_size{0.05f};
        float3 origin{-3.2f, -0.8f, -3.2f};
        int3 resolution{128, 192, 128};
        float cfl{0.75f};
        float max_time_step{1.0f / 48.0f};
        float3 wind{};
        ForcesConfig forces;
        PressureConfig pressure;
    };

    struct Solver {
        ::cuda::device_ref device;
        ::cuda::device_memory_pool pool;
        ::cuda::stream stream;

        ::geometry::Scene scene;
        const Config config;

        Statistics statistics;
        double time{};

        std::vector<Source> sources;
        std::vector<Collider> colliders;

        Grid grid;
        Pressure pressure;

        explicit Solver(const Config& settings = {});
        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        ~Solver() noexcept;

        template <bool profiling = false>
        void advance(double duration);
        void synchronize();
        void checkpoint(const char* path, bool restore);
        std::span<const float> download_density();

    private:
        ::cuda::pinned_memory_pool host_pool;

        std::vector<SourceRegion> source_regions;
        float* maximum{};
        ::cuda::buffer<float, ::cuda::mr::device_accessible> field_storage;
        ::cuda::buffer<unsigned char, ::cuda::mr::device_accessible> mask_storage;
        ::cuda::buffer<SourceView, ::cuda::mr::device_accessible> source_storage;
        ::cuda::buffer<ColliderView, ::cuda::mr::device_accessible> collider_storage;
        std::vector<::cuda::buffer<float, ::cuda::mr::device_accessible>> source_weight_storage;

        ::cuda::buffer<float, ::cuda::mr::host_accessible> host_output;
        std::vector<std::array<::cuda::timed_event, 2>> events;

        std::vector<Source> source_inputs;
        std::vector<Collider> collider_inputs;
        std::vector<::geometry::InstanceView> source_instances;
        std::vector<::geometry::InstanceView> collider_instances;
        bool geometry_dirty{true};

        void prepare_inputs(double duration);
    };
}
#endif
