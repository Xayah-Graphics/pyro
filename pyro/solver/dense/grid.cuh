#ifndef PYRO_SOLVER_DENSE_GRID_CUH
#define PYRO_SOLVER_DENSE_GRID_CUH

#include <cuda/stream>

#include <solver/common/scene.cuh>

namespace pyro::dense {
    inline constexpr int tile_cells = 1024, threads = 256;

    struct Layout {
        int3 size{};
        int pitch{};
        int slice{};
        int cells{};
        int blocks{};
    };

    struct Fields {
        float* velocity[3]{};
        float* smoke{};
        float* temperature{};
    };

    struct Grid {
        Layout layout;
        float3 origin{};
        int3 geometry_lower{};
        int3 geometry_extent{};
        unsigned char* fluid{};
        float* aperture[3]{};
        float* solid_velocity[3]{};
        float* solid_divergence{};
        float* distance[3]{};
        float geometry_time{};
        float geometry_duration{};
        float collider_speed{};
        BoundaryMode boundary_mode{BoundaryMode::none};
        Fields state;
        Fields midpoint;
        Fields next;
        Fields transported;
        Fields upper;
    };

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

    template <bool bounds = false>
    __device__ inline float sample(const Layout& layout, const float* field, float3 position, float background, float* lower = nullptr, float* upper = nullptr) {
        int3 p{static_cast<int>(floorf(position.x)), static_cast<int>(floorf(position.y)), static_cast<int>(floorf(position.z))};
        float3 t{position.x - p.x, position.y - p.y, position.z - p.z};
        float result = 0.0f, lo = 1.0e30f, hi = -1.0e30f;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    float v = value(field, cell_index(layout, {p.x + x, p.y + y, p.z + z}), background);
                    result += v * (x ? t.x : 1.0f - t.x) * (y ? t.y : 1.0f - t.y) * (z ? t.z : 1.0f - t.z);
                    if constexpr (bounds) {
                        lo = fminf(lo, v);
                        hi = fmaxf(hi, v);
                    }
                }
        if constexpr (bounds) *lower = lo;
        if constexpr (bounds) *upper = hi;
        return result;
    }

    __device__ inline float3 sample_velocity(const Layout& layout, Fields fields, float3 p, float3 wind) {
        return {sample(layout, fields.velocity[0], {p.x, p.y - 0.5f, p.z - 0.5f}, wind.x), sample(layout, fields.velocity[1], {p.x - 0.5f, p.y, p.z - 0.5f}, wind.y), sample(layout, fields.velocity[2], {p.x - 0.5f, p.y - 0.5f, p.z}, wind.z)};
    }

    template <BoundaryMode boundary_mode>
    __device__ inline float sample_boundary(Grid grid, float3 world, float time, float h) {
        float inverse_h = 1.0f / h;
        float3 position{(world.x - grid.origin.x) * inverse_h, (world.y - grid.origin.y) * inverse_h, (world.z - grid.origin.z) * inverse_h};
        if constexpr (boundary_mode == BoundaryMode::stationary) return sample(grid.layout, grid.distance[0], position, 4.0f * h);
        float fraction = fminf(2.0f, fmaxf(0.0f, 2.0f * (time - grid.geometry_time) / grid.geometry_duration));
        int lower      = fraction < 1.0f ? 0 : 1;
        float weight   = fraction - lower;
        float a        = sample(grid.layout, grid.distance[lower], position, 4.0f * h);
        float b        = sample(grid.layout, grid.distance[lower + 1], position, 4.0f * h);
        return a + (b - a) * weight;
    }

    void initialize_grid(Grid grid, float3 wind, ::cuda::stream_ref stream);

    void sample_geometry(Grid grid, float h, const ColliderView* colliders, int count, float time, int sample, ::cuda::stream_ref stream);

    void update_boundary(Grid grid, float h, const ColliderView* colliders, int count, float time, int sample, ::cuda::stream_ref stream);
}
#endif
