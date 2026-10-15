#ifndef PYRO_LFM_CUDA_CUH
#define PYRO_LFM_CUDA_CUH
#include "../geometry.cuh"
#include "../pressure.cuh"

namespace pyro::cuda::lfm {
    inline constexpr int tileCells = 512, blockThreads = 128;
    struct Config {
        float cellSize{1.0f / 128};
        float3 origin{};
        int3 resolution{256, 128, 128};
        float3 wind{0.5638156f, 0.2052121f, 0};
        int cycleSteps{5};
        float pressureRelativeTolerance{1.e-5f}, divergenceTolerance{2.e-4f};
        int pressureMaxIterations{200};
    };
    struct Layout {
        int3 size{}, tiles{};
        int blocks{}, count{};
    };
    struct Velocity {
        float* axis[3]{};
    };
    struct Map {
        float3* position[3]{};
        float3* tangent[3]{}; // Column of the map Jacobian for this MAC face axis.
    };
    struct Grid {
        Layout cells, faces[3];
        unsigned char* fluid{};
        unsigned char* boundary[3]{}; // 0: fluid face, 1: solid, 2: prescribed domain face.
        Velocity initial, work, impulse, error;
        Velocity* history{};
        Map backward, forward;
        float* output{};
    };
    struct PressureLevel {
        Layout layout;
        float regularWeight{1};
        float *diagonal{}, *weight[3]{};
        float *solution{}, *temporary{}, *rhs{};
        unsigned char* kind{}; // 0: empty, 1: constant stencil including halo, 2: general.
    };
    struct PressureWorkspace {
        PressureLevel* levels{};
        int levelCount{};
        int* parents{};
        int *componentCounts{}, *blockComponents{};
        double* componentSums{};
        double* solution{};
        float *residual{}, *direction{}, *product{};
        double* reductions{};
        IterationState* iteration{};
        cudaGraphExec_t graph{};
    };

    void initialize(Grid grid, Config config, cudaStream_t stream);
    void geometry(Grid grid, Config config, const Collider* colliders, int count, cudaStream_t stream);
    void enforce_boundary(Grid grid, Velocity velocity, float3 wind, cudaStream_t stream);
    void advect(Grid grid, Velocity destination, Velocity original, Velocity carrier, float dtOverH, float3 wind, cudaStream_t stream);
    void reconstruct(Grid grid, Config config, float dt, cudaStream_t stream);
    void measure(Grid grid, Velocity velocity, float h, double* reductions, cudaStream_t stream);
    void export_vorticity(Grid grid, Velocity velocity, float h, cudaStream_t stream);
    void build_pressure(Grid grid, PressureWorkspace workspace, cudaStream_t stream);
    void begin_pressure(Grid grid, Velocity velocity, PressureWorkspace workspace, Config config, cudaStream_t stream);
    void iterate_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream);
    void check_pressure(Grid grid, Velocity velocity, PressureWorkspace workspace, Config config, cudaStream_t stream);
    void apply_pressure(Grid grid, Velocity velocity, PressureWorkspace workspace, cudaStream_t stream);

#ifdef __CUDACC__
    __host__ __device__ inline bool contains(Layout layout, int3 p) {
        return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < layout.size.x && p.y < layout.size.y && p.z < layout.size.z;
    }
    __host__ __device__ inline int index(Layout layout, int3 p) {
        return (((p.z >> 3) * layout.tiles.y + (p.y >> 3)) * layout.tiles.x + (p.x >> 3)) * tileCells + ((p.z & 7) << 6) + ((p.y & 7) << 3) + (p.x & 7);
    }
    __device__ inline int3 coordinate(Layout layout, int i) {
        int tile = i >> 9;
        return {(tile % layout.tiles.x) * 8 + (i & 7), (tile / layout.tiles.x % layout.tiles.y) * 8 + ((i >> 3) & 7), (tile / (layout.tiles.x * layout.tiles.y)) * 8 + ((i >> 6) & 7)};
    }
    __device__ inline float read(Layout layout, const float* values, int3 p) {
        return contains(layout, p) ? values[index(layout, p)] : 0;
    }
    __device__ inline float3 face_position(int3 p, int axis) {
        return {p.x + (axis != 0 ? .5f : 0), p.y + (axis != 1 ? .5f : 0), p.z + (axis != 2 ? .5f : 0)};
    }
#endif
} // namespace pyro::cuda::lfm
#endif
