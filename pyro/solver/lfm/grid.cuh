#ifndef PYRO_SOLVER_LFM_GRID_CUH
#define PYRO_SOLVER_LFM_GRID_CUH

#include <cuda/stream>
#include <cuda_runtime.h>

namespace pyro {
    struct ColliderView;
}

namespace pyro::lfm {
    inline constexpr int tile_cells = 512, block_threads = 128;

    struct Layout {
        int3 size{};
        int3 tiles{};
        int blocks{};
        int count{};
    };

    struct Velocity {
        float* axis[3]{};
    };

    struct Map {
        float3* position[3]{};
        float3* tangent[3]{};
    };

    struct Grid {
        Layout cells;
        Layout faces[3];
        unsigned char* fluid{};
        unsigned char* boundary[3]{};
        Velocity initial;
        Velocity work;
        Velocity impulse;
        Velocity error;
        Velocity* history{};
        Map backward;
        Map forward;
        float* output{};
    };

    __host__ inline Layout make_layout(int3 size) {
        Layout result;
        result.size   = size;
        result.tiles  = {(size.x + 7) / 8, (size.y + 7) / 8, (size.z + 7) / 8};
        result.blocks = result.tiles.x * result.tiles.y * result.tiles.z;
        result.count  = result.blocks * tile_cells;
        return result;
    }

    __host__ __device__ inline bool contains(Layout layout, int3 p) {
        return p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < layout.size.x && p.y < layout.size.y && p.z < layout.size.z;
    }

    __host__ __device__ inline int index(Layout layout, int3 p) {
        return (((p.z >> 3) * layout.tiles.y + (p.y >> 3)) * layout.tiles.x + (p.x >> 3)) * tile_cells + ((p.z & 7) << 6) + ((p.y & 7) << 3) + (p.x & 7);
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

    template <bool gradient>
    __device__ inline float sample_axis(Layout layout, const float* field, float3 position, int axis, float3& derivative) {
        float q[3]{position.x - (axis != 0 ? .5f : 0), position.y - (axis != 1 ? .5f : 0), position.z - (axis != 2 ? .5f : 0)};
        int dimensions[3]{layout.size.x, layout.size.y, layout.size.z};
        int address[3][3];
        float weights[3][3], slopes[3][3];
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            int center    = static_cast<int>(floorf(q[a] + .5f));
            float t       = q[a] - center;
            weights[a][0] = .5f * (.5f - t) * (.5f - t);
            weights[a][1] = .75f - t * t;
            weights[a][2] = .5f * (.5f + t) * (.5f + t);
            if constexpr (gradient) {
                slopes[a][0] = t - .5f;
                slopes[a][1] = -2 * t;
                slopes[a][2] = t + .5f;
            }
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                int c         = min(max(center + k - 1, 0), dimensions[a] - 1);
                address[a][k] = a == 0 ? (c >> 3) * 512 + (c & 7) : a == 1 ? (c >> 3) * layout.tiles.x * 512 + (c & 7) * 8 : (c >> 3) * layout.tiles.x * layout.tiles.y * 512 + (c & 7) * 64;
            }
        }
        float reference = field[address[0][1] + address[1][1] + address[2][1]];
        float value     = 0;
        derivative      = {};
#pragma unroll
        for (int z = 0; z < 3; ++z)
#pragma unroll
            for (int y = 0; y < 3; ++y)
#pragma unroll
                for (int x = 0; x < 3; ++x) {
                    float v = field[address[0][x] + address[1][y] + address[2][z]] - reference;
                    value += v * weights[0][x] * weights[1][y] * weights[2][z];
                    if constexpr (gradient) {
                        derivative.x += v * slopes[0][x] * weights[1][y] * weights[2][z];
                        derivative.y += v * weights[0][x] * slopes[1][y] * weights[2][z];
                        derivative.z += v * weights[0][x] * weights[1][y] * slopes[2][z];
                    }
                }
        return reference + value;
    }

    __device__ inline float3 sample_velocity(Grid grid, Velocity velocity, float3 p) {
        float3 unused;
        return {sample_axis<false>(grid.faces[0], velocity.axis[0], p, 0, unused), sample_axis<false>(grid.faces[1], velocity.axis[1], p, 1, unused), sample_axis<false>(grid.faces[2], velocity.axis[2], p, 2, unused)};
    }

    __device__ inline float3 sample_tangent(Grid grid, Velocity velocity, float3 p, float3 tangent, float3& derivative) {
        float3 dx, dy, dz;
        float3 u{sample_axis<true>(grid.faces[0], velocity.axis[0], p, 0, dx), sample_axis<true>(grid.faces[1], velocity.axis[1], p, 1, dy), sample_axis<true>(grid.faces[2], velocity.axis[2], p, 2, dz)};
        derivative = {dx.x * tangent.x + dx.y * tangent.y + dx.z * tangent.z, dy.x * tangent.x + dy.y * tangent.y + dy.z * tangent.z, dz.x * tangent.x + dz.y * tangent.y + dz.z * tangent.z};
        return u;
    }

    void initialize(Grid grid, float3 wind, ::cuda::stream_ref stream);

    void update_geometry(Grid grid, float cell_size, float3 origin, const ColliderView* colliders, int count, ::cuda::stream_ref stream);

    void enforce_boundary(Grid grid, Velocity velocity, float3 wind, ::cuda::stream_ref stream);
}
#endif
