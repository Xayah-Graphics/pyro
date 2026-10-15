#include "sampling.cuh"

namespace pyro::cuda::lfm {
    template <int axis, bool backward>
    __global__ void march_kernel(Grid grid, int steps, float dtOverH) {
        Layout layout = grid.faces[axis];
        Map map       = backward ? grid.backward : grid.forward;
        float step    = backward ? -dtOverH : dtOverH;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i     = blockIdx.x * tileCells + local;
            int3 cell = coordinate(layout, i);
            if (!contains(layout, cell)) continue;
            float3 p = face_position(cell, axis);
            float3 t{axis == 0 ? 1.f : 0.f, axis == 1 ? 1.f : 0.f, axis == 2 ? 1.f : 0.f};
            // Keep each complete trajectory in registers. Intermediate maps are
            // not consumed elsewhere, so no per-segment global map writes are needed.
            for (int k = 0; k < steps; ++k) {
                Velocity velocity = grid.history[backward ? steps - 1 - k : k];
                float3 j1, j2, j3, j4;
                float3 v1 = sample_tangent(grid, velocity, p, t, j1);
                float3 v2 = sample_tangent(grid, velocity, p + v1 * (.5f * step), t + j1 * (.5f * step), j2);
                float3 v3 = sample_tangent(grid, velocity, p + v2 * (.5f * step), t + j2 * (.5f * step), j3);
                float3 v4 = sample_tangent(grid, velocity, p + v3 * step, t + j3 * step, j4);
                p         = p + (v1 + v2 * 2 + v3 * 2 + v4) * (step / 6);
                t         = t + (j1 + j2 * 2 + j3 * 2 + j4) * (step / 6);
            }
            map.position[axis][i] = p;
            map.tangent[axis][i]  = t;
        }
    }
    template <int axis, int pass>
    __global__ void pullback_kernel(Grid grid, float3 wind) {
        Layout layout        = grid.faces[axis];
        Map map              = pass == 1 ? grid.forward : grid.backward;
        Velocity source      = pass == 0 ? grid.initial : pass == 1 ? grid.impulse : grid.error;
        Velocity destination = pass == 0 ? grid.impulse : pass == 1 ? grid.error : grid.work;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i       = blockIdx.x * tileCells + local;
            int3 p      = coordinate(layout, i);
            float value = 0;
            if (contains(layout, p)) {
                value = dot(map.tangent[axis][i], sample_velocity(grid, source, map.position[axis][i]));
                if constexpr (pass == 1) value = .5f * (value - grid.initial.axis[axis][i]);
                if constexpr (pass == 2) {
                    value       = grid.impulse.axis[axis][i] - value;
                    float lower = 1.e30f, upper = -1.e30f;
                    for (int a = 0; a < 3; ++a)
                        for (int sign = -1; sign <= 1; sign += 2) {
                            int3 q = offset(p, a, sign);
                            if (!contains(layout, q)) continue;
                            float neighbor = grid.impulse.axis[axis][index(layout, q)];
                            lower          = fminf(lower, neighbor);
                            upper          = fmaxf(upper, neighbor);
                        }
                    value        = fminf(upper, fmaxf(lower, value));
                    int boundary = grid.boundary[axis][i];
                    if (boundary) value = boundary == 2 ? component(wind, axis) : 0;
                }
            }
            destination.axis[axis][i] = value;
        }
    }
    void reconstruct(Grid grid, Config config, float dt, cudaStream_t stream) {
        float step = dt / config.cellSize;
        march_kernel<0, true><<<grid.faces[0].blocks, blockThreads, 0, stream>>>(grid, config.cycleSteps, step);
        march_kernel<1, true><<<grid.faces[1].blocks, blockThreads, 0, stream>>>(grid, config.cycleSteps, step);
        march_kernel<2, true><<<grid.faces[2].blocks, blockThreads, 0, stream>>>(grid, config.cycleSteps, step);
        march_kernel<0, false><<<grid.faces[0].blocks, blockThreads, 0, stream>>>(grid, config.cycleSteps, step);
        march_kernel<1, false><<<grid.faces[1].blocks, blockThreads, 0, stream>>>(grid, config.cycleSteps, step);
        march_kernel<2, false><<<grid.faces[2].blocks, blockThreads, 0, stream>>>(grid, config.cycleSteps, step);
        pullback_kernel<0, 0><<<grid.faces[0].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<1, 0><<<grid.faces[1].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<2, 0><<<grid.faces[2].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<0, 1><<<grid.faces[0].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<1, 1><<<grid.faces[1].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<2, 1><<<grid.faces[2].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<0, 2><<<grid.faces[0].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<1, 2><<<grid.faces[1].blocks, blockThreads, 0, stream>>>(grid, config.wind);
        pullback_kernel<2, 2><<<grid.faces[2].blocks, blockThreads, 0, stream>>>(grid, config.wind);
    }
} // namespace pyro::cuda::lfm
