#ifndef PYRO_SOLVER_LFM_SOLVER_CUH
#define PYRO_SOLVER_LFM_SOLVER_CUH

#include <array>
#include <span>
#include <vector>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#include "grid.cuh"
#include "pressure.cuh"
#include "statistics.cuh"
#include <geometry/geometry.cuh>
#include <solver/common/scene.cuh>

namespace pyro::lfm {
    struct Config {
        float cell_size{1.0f / 128};
        float3 origin{};
        int3 resolution{256, 128, 128};
        float3 wind{0.5638156f, 0.2052121f, 0};
        int cycle_steps{5};
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
        std::span<const float> download_vorticity();

    private:
        ::cuda::pinned_memory_pool host_pool;

        std::vector<Velocity> history;
        ::cuda::buffer<unsigned char, ::cuda::mr::device_accessible> mask_storage;
        ::cuda::buffer<Velocity, ::cuda::mr::device_accessible> history_storage;
        ::cuda::buffer<float, ::cuda::mr::device_accessible> output_storage;
        ::cuda::buffer<ColliderView, ::cuda::mr::device_accessible> collider_storage;
        std::vector<::cuda::buffer<float, ::cuda::mr::device_accessible>> face_storage;
        std::vector<::cuda::buffer<float3, ::cuda::mr::device_accessible>> map_storage;

        ::cuda::buffer<float, ::cuda::mr::host_accessible> host_output;
        std::vector<std::array<::cuda::timed_event, 2>> events;

        std::vector<Collider> collider_inputs;
        std::vector<::geometry::InstanceView> collider_instances;
        bool geometry_dirty{true};

        void prepare_inputs(double duration);
        void prepare_geometry();
    };
}
#endif
