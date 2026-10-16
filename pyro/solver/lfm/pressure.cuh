#ifndef PYRO_SOLVER_LFM_PRESSURE_CUH
#define PYRO_SOLVER_LFM_PRESSURE_CUH

#include <memory>
#include <type_traits>
#include <vector>

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include "grid.cuh"

namespace pyro::lfm {
    struct PressureConfig {
        float relative_tolerance{1.e-5f};
        float divergence_tolerance{2.e-4f};
        int max_iterations{200};
    };

    struct PressureLevel {
        Layout layout;
        float regular_weight{1};
        float* diagonal{};
        float* weight[3]{};
        float* solution{};
        float* temporary{};
        float* rhs{};
        unsigned char* kind{};
    };

    struct IterationState {
        double rz{};
        double previous_rz{};
        double denominator{};
        double initial_norm{};
        float max_residual{};
        int iterations{};
        int converged{};
        int failed{};
    };

    struct PressureView {
        int level_count{};
        int* parents{};
        int* component_counts{};
        int* block_components{};
        double* component_sums{};
        double* solution{};
        float* residual{};
        float* direction{};
        float* product{};
        double* reductions{};
        IterationState* iteration{};
    };

    struct Pressure {
        const PressureConfig config;
        PressureView view;

        Pressure(int3 resolution, PressureConfig settings, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool);

        void build(Grid grid);
        int project(Grid grid, Velocity velocity, float h);

    private:
        ::cuda::stream_ref stream;
        ::cuda::device_memory_pool_ref pool;

        std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> iteration_graph;
        ::cuda::buffer<IterationState, ::cuda::mr::device_accessible> iteration_storage;

        ::cuda::buffer<float, ::cuda::mr::device_accessible> pressure_storage;
        ::cuda::buffer<double, ::cuda::mr::device_accessible> double_storage;
        ::cuda::buffer<int, ::cuda::mr::device_accessible> index_storage;
        std::vector<PressureLevel> hierarchy;
        std::vector<::cuda::buffer<float, ::cuda::mr::device_accessible>> level_storage;
        std::vector<::cuda::buffer<unsigned char, ::cuda::mr::device_accessible>> level_kind_storage;
    };
}
#endif
