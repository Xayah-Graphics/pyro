#ifndef PYRO_SOLVER_SPARSE_SOLVER_CUH
#define PYRO_SOLVER_SPARSE_SOLVER_CUH

#include <array>
#include <cstddef>
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

namespace pyro::sparse {
    struct Config {
        int initial_blocks{4096};
        GridConfig grid;
        float max_time_step{1.f / 48};
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
        Scratch scratch;
        Pressure pressure;

        explicit Solver(const Config& settings = {});
        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        ~Solver() noexcept;

        template <bool profiling = false>
        void advance(double duration);
        void synchronize();
        void checkpoint(const char* path, bool restore);
        std::span<const DensityBlock> download_density();

    private:
        ::cuda::pinned_memory_pool host_pool;

        Grid spare;
        Topology topology;
        ::cuda::buffer<Block, ::cuda::mr::device_accessible> block_storage;
        ::cuda::buffer<unsigned long long, ::cuda::mr::device_accessible> key_storage;
        ::cuda::buffer<int, ::cuda::mr::device_accessible> index_storage;
        ::cuda::buffer<Face, ::cuda::mr::device_accessible> face_storage;
        ::cuda::buffer<Links, ::cuda::mr::device_accessible> link_storage;
        ::cuda::buffer<unsigned char, ::cuda::mr::device_accessible> mask_storage;
        ::cuda::buffer<float, ::cuda::mr::device_accessible> field_storage;
        ::cuda::buffer<std::byte, ::cuda::mr::device_accessible> sort_storage;
        ::cuda::buffer<SourceView, ::cuda::mr::device_accessible> source_storage;
        ::cuda::buffer<ColliderView, ::cuda::mr::device_accessible> collider_storage;
        ::cuda::buffer<float, ::cuda::mr::device_accessible> source_weight_storage;

        ::cuda::buffer<DensityBlock, ::cuda::mr::host_accessible> host_output;
        ::cuda::buffer<DensityBlock, ::cuda::mr::device_accessible> density_storage;
        std::vector<std::array<::cuda::timed_event, 2>> events;

        std::vector<Source> source_inputs;
        std::vector<Collider> collider_inputs;
        std::vector<::geometry::InstanceView> source_instances;
        std::vector<::geometry::InstanceView> collider_instances;
        bool geometry_dirty{true};
        bool source_weights_dirty{true};
        int substep{};
        bool topology_dirty{true};

        void prepare_inputs(double duration);
        void reserve(int required);
        bool adapt(float time);
    };
}
#endif
