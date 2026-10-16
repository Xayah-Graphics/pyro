#ifndef PYRO_LFM_SAMPLING_CUH
#define PYRO_LFM_SAMPLING_CUH
#include "cuda.cuh"

namespace pyro::cuda::lfm {
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
} // namespace pyro::cuda::lfm
#endif
