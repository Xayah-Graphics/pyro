#ifndef PYRO_SOLVER_DENSE_PRESSURE_CUH
#define PYRO_SOLVER_DENSE_PRESSURE_CUH

#include <memory>
#include <type_traits>
#include <vector>

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include "grid.cuh"

namespace pyro::dense {
    struct PressureConfig {
        float relative_tolerance{1.e-5f};
        float divergence_tolerance{2.e-4f};
        int max_iterations{200};
    };

    struct PressureLevel {
        Layout layout;
        int3 coarsening{2, 2, 2};
        float constant_coefficient{};
        float* diagonal{};
        float* boundary{};
        float* face[3]{};
        float* solution{};
        float* rhs{};
        float* temporary{};
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
        int anchor_count{};
        float* direction{};
        float* product{};
        double* solution{};
        double* rhs{};
        double* closed_residual{};
        double* closed_product{};
        float* residual{};
        int* parents{};
        int* anchor_indices{};
        unsigned char* anchors{};
        double* reductions{};
        IterationState* iteration{};
    };

    struct Pressure {
        const PressureConfig config;
        PressureView view;
        bool matrix_dirty{true};
        std::vector<PressureLevel> hierarchy;

        Pressure(int3 resolution, PressureConfig settings, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool);

        void build(Grid grid, float h);
        int project(Grid grid, Fields fields, float h, float3 wind);

    private:
        ::cuda::stream_ref stream;

        std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> iteration_graph;
        ::cuda::buffer<IterationState, ::cuda::mr::device_accessible> iteration_storage;

        ::cuda::buffer<int, ::cuda::mr::device_accessible> index_storage;
        ::cuda::buffer<double, ::cuda::mr::device_accessible> reduction_storage;
        ::cuda::buffer<double, ::cuda::mr::device_accessible> closed_storage;
        std::vector<::cuda::buffer<float, ::cuda::mr::device_accessible>> level_storage;
    };
}
#endif
