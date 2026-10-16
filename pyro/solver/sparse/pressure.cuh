#ifndef PYRO_SOLVER_SPARSE_PRESSURE_CUH
#define PYRO_SOLVER_SPARSE_PRESSURE_CUH

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include "grid.cuh"

namespace pyro::sparse {
    struct PressureConfig {
        float relative_tolerance{1.e-5f};
        float divergence_tolerance{2.e-4f};
        int max_iterations{300};
    };

    struct Edge {
        int left{};
        int right{};
        float weight{};
    };

    struct PressureLevel {
        Node* nodes{};
        int* parent{};
        int* children{};
        Edge* edges{};
        float* face_weights{};
        int* adjacency{};
        float* boundary{};
        float* diagonal{};
        float* rhs{};
        float* solution{};
        float* temporary{};
        int count{};
        int edge_count{};
        int capacity{};
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
        double* solution{};
        float* residual{};
        float* direction{};
        float* product{};
        int* components{};
        int* open{};
        unsigned char* anchors{};
        double* reductions{};
        double* component_sums{};
        bool closed{};
        IterationState* iteration{};
    };

    struct Pressure {
        const PressureConfig config;
        PressureView view;
        bool matrix_dirty{true};

        Pressure(PressureConfig settings, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool);

        void reserve(Layout layout);
        void build(Grid grid, Scratch scratch);
        int project(Grid grid, Fields fields, Scratch scratch);

    private:
        ::cuda::stream_ref stream;
        ::cuda::device_memory_pool_ref pool;

        std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>, decltype(&cudaGraphExecDestroy)> iteration_graph;
        ::cuda::buffer<IterationState, ::cuda::mr::device_accessible> iteration_storage;

        std::array<PressureLevel, 32> hierarchy{};
        ::cuda::buffer<Node, ::cuda::mr::device_accessible> node_storage;
        ::cuda::buffer<Edge, ::cuda::mr::device_accessible> edge_storage;
        ::cuda::buffer<int, ::cuda::mr::device_accessible> index_storage;
        ::cuda::buffer<unsigned char, ::cuda::mr::device_accessible> anchor_storage;
        ::cuda::buffer<float, ::cuda::mr::device_accessible> field_storage;
        ::cuda::buffer<double, ::cuda::mr::device_accessible> double_storage;
        std::vector<::cuda::buffer<Node, ::cuda::mr::device_accessible>> coarse_node_storage;
        std::vector<::cuda::buffer<Edge, ::cuda::mr::device_accessible>> coarse_edge_storage;
        std::vector<::cuda::buffer<int, ::cuda::mr::device_accessible>> coarse_index_storage;
        std::vector<::cuda::buffer<float, ::cuda::mr::device_accessible>> coarse_field_storage;
    };
}
#endif
