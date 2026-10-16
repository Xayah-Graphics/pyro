#ifndef PYRO_DENSE_CUDA_CUH
#define PYRO_DENSE_CUDA_CUH
#include "../geometry.cuh"
namespace pyro::cuda::dense {
    inline constexpr int tile_cells = 1024, threads = 256;
    struct Config {
        float cell_size{0.05f};
        float3 origin{-3.2f, -0.8f, -3.2f};
        int3 resolution{128, 192, 128};
        float cfl{0.75f};
        float max_time_step{1.0f / 48.0f};
        float buoyancy{1.0f};
        float smoke_weight{0.05f};
        float3 up{0.0f, 1.0f, 0.0f};
        float3 wind{};
        float cooling{0.1f};
        float dissipation{0.01f};
        float pressure_relative_tolerance{1.0e-5f};
        float divergence_tolerance{2.0e-4f};
        int pressure_max_iterations{200};
    };


    struct SourceRegion {
        int3 lower{}, extent{};
        float* weights{};
        int capacity{};
        bool cached{};
    };
    struct Fields {
        float* velocity[3]{};
        float* smoke{};
        float* temperature{};
    };
    struct Layout {
        int3 size{};
        int pitch{}, slice{}, cells{}, blocks{};
    };
    struct Grid {
        Layout layout;
        float3 origin{};
        int3 geometry_lower{}, geometry_extent{};
        unsigned char* fluid{};
        float* aperture[3]{};
        float* solid_velocity[3]{};
        float* solid_divergence{};
        float* distance[3]{};
        float geometry_time{}, geometry_duration{}, collider_speed{};
        bool static_geometry{true};
        Fields state, midpoint, next, transported;
    };
    struct PressureLevel {
        Layout layout;
        int3 coarsening{2, 2, 2};
        float* diagonal{};
        float* boundary{};
        float* face[3]{};
        float* solution{};
        float* rhs{};
        float* temporary{};
    };

    struct IterationState {
        double rz{}, previous_rz{}, denominator{}, initial_norm{};
        float max_residual{};
        int iterations{}, converged{}, failed{};
    };

    struct PressureWorkspace {
        PressureLevel* levels{};
        int level_count{};
        float* direction{};
        float* product{};
        float* solution{};
        float* residual{};
        int* parents{};
        int* open_components{};
        unsigned char* anchors{};
        double* reductions{};
        IterationState* iteration{};
        cudaGraphExec_t graph{};
    };

    void initialize_grid(Grid grid, float3 wind, cudaStream_t stream);
    void sample_geometry(Grid grid, float h, const Collider* colliders, int count, float time, int sample, cudaStream_t stream);
    void update_boundary(Grid grid, float h, const Collider* colliders, int count, float time, int sample, cudaStream_t stream);
    void advect(Grid grid, Config config, Fields destination, Fields original, Fields carrier, bool obstacles, float start_time, float dt, cudaStream_t stream);
    void cache_source(Grid grid, Config config, const Source* sources, int index, SourceRegion region, cudaStream_t stream);
    void apply_sources_forces(Grid grid, Config config, Fields destination, Fields force_fields, const Source* sources, const SourceRegion* regions, int count, float time, float dt, cudaStream_t stream);
    void measure(Grid grid, Fields fields, Config config, const Source* sources, int source_count, float time, double* reductions, float* maximum, bool full, cudaStream_t stream);
    void assemble_pressure(Grid grid, Fields fields, PressureWorkspace workspace, float h, bool obstacles, bool rebuild, cudaStream_t stream);
    void coarse_matrix(PressureLevel fine, PressureLevel coarse, cudaStream_t stream);
    void begin_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream);
    void iterate_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream);
    void check_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream);
    void apply_pressure(Grid grid, Fields fields, const float* pressure, float h, float3 wind, cudaStream_t stream);
#ifdef __CUDACC__

    __host__ __device__ inline bool interior(Layout layout, int3 p) {
        return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < layout.size.x && p.y < layout.size.y && p.z < layout.size.z;
    }
    __device__ inline int3 cell_coordinate(Layout layout, int i) {
        int z         = i / layout.slice;
        int remainder = i - z * layout.slice;
        int y         = remainder / layout.pitch;
        return {remainder - y * layout.pitch - 1, y - 1, z - 1};
    }
    __device__ inline int cell_index(Layout layout, int3 p) {
        if (p.x < -1 || p.y < -1 || p.z < -1 || p.x > layout.size.x || p.y > layout.size.y || p.z > layout.size.z) return -1;
        return p.x + 1 + (p.y + 1) * layout.pitch + (p.z + 1) * layout.slice;
    }
    __device__ inline int neighbor(Layout layout, int i, int axis, int sign) {
        int stride = axis == 0 ? 1 : axis == 1 ? layout.pitch : layout.slice;
        int j      = i + sign * stride;
        return j >= 0 && j < layout.cells ? j : -1;
    }
    __device__ inline bool face_valid(Layout layout, int3 p, int axis) {
        return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < layout.size.x + (axis == 0) && p.y < layout.size.y + (axis == 1) && p.z < layout.size.z + (axis == 2);
    }
    __device__ inline float value(const float* field, int index, float background = 0.0f) {
        return index < 0 ? background : field[index];
    }
    __device__ inline float sample(const Layout& layout, const float* field, float3 position, float background, float* lower = nullptr, float* upper = nullptr) {
        int3 p{static_cast<int>(floorf(position.x)), static_cast<int>(floorf(position.y)), static_cast<int>(floorf(position.z))};
        float3 t{position.x - p.x, position.y - p.y, position.z - p.z};
        float result = 0.0f, lo = 1.0e30f, hi = -1.0e30f;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    float v = value(field, cell_index(layout, {p.x + x, p.y + y, p.z + z}), background);
                    result += v * (x ? t.x : 1.0f - t.x) * (y ? t.y : 1.0f - t.y) * (z ? t.z : 1.0f - t.z);
                    lo = fminf(lo, v);
                    hi = fmaxf(hi, v);
                }
        if (lower) *lower = lo;
        if (upper) *upper = hi;
        return result;
    }
    __device__ inline float3 sample_velocity(const Layout& layout, Fields fields, float3 p, float3 wind) {
        return {sample(layout, fields.velocity[0], {p.x, p.y - 0.5f, p.z - 0.5f}, wind.x), sample(layout, fields.velocity[1], {p.x - 0.5f, p.y, p.z - 0.5f}, wind.y), sample(layout, fields.velocity[2], {p.x - 0.5f, p.y - 0.5f, p.z}, wind.z)};
    }

    __device__ inline float sample_boundary(Grid grid, float3 world, float time, float h) {
        float inverse_h = 1.0f / h;
        float3 position{(world.x - grid.origin.x) * inverse_h, (world.y - grid.origin.y) * inverse_h, (world.z - grid.origin.z) * inverse_h};
        if (grid.static_geometry) return sample(grid.layout, grid.distance[0], position, 4.0f * h);
        float fraction = fminf(2.0f, fmaxf(0.0f, 2.0f * (time - grid.geometry_time) / grid.geometry_duration));
        int lower      = fraction < 1.0f ? 0 : 1;
        float weight   = fraction - lower;
        float a        = sample(grid.layout, grid.distance[lower], position, 4.0f * h);
        float b        = sample(grid.layout, grid.distance[lower + 1], position, 4.0f * h);
        return a + (b - a) * weight;
    }

#endif
} // namespace pyro::cuda::dense
#endif
