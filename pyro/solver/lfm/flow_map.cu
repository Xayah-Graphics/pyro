#include "flow_map.cuh"
#include <cuda/launch>

namespace pyro::lfm {
    template <int axis, bool backward>
    __global__ void march_kernel(Grid grid, int steps, float dt_over_h) {
        Layout layout = grid.faces[axis];
        Map map       = backward ? grid.backward : grid.forward;
        float step    = backward ? -dt_over_h : dt_over_h;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i     = blockIdx.x * tile_cells + local;
            int3 cell = coordinate(layout, i);
            if (!contains(layout, cell)) continue;
            float3 p = face_position(cell, axis);
            float3 t{axis == 0 ? 1.f : 0.f, axis == 1 ? 1.f : 0.f, axis == 2 ? 1.f : 0.f};
            for (int k = 0; k < steps; ++k) {
                Velocity velocity = grid.history[backward ? steps - 1 - k : k];
                float3 j1, j2, j3, j4;
                float3 v1 = sample_tangent(grid, velocity, p, t, j1);
                float3 v2 = sample_tangent(grid, velocity, {p.x + v1.x * (.5f * step), p.y + v1.y * (.5f * step), p.z + v1.z * (.5f * step)}, {t.x + j1.x * (.5f * step), t.y + j1.y * (.5f * step), t.z + j1.z * (.5f * step)}, j2);
                float3 v3 = sample_tangent(grid, velocity, {p.x + v2.x * (.5f * step), p.y + v2.y * (.5f * step), p.z + v2.z * (.5f * step)}, {t.x + j2.x * (.5f * step), t.y + j2.y * (.5f * step), t.z + j2.z * (.5f * step)}, j3);
                float3 v4 = sample_tangent(grid, velocity, {p.x + v3.x * step, p.y + v3.y * step, p.z + v3.z * step}, {t.x + j3.x * step, t.y + j3.y * step, t.z + j3.z * step}, j4);
                p         = {p.x + (v1.x + v2.x * 2 + v3.x * 2 + v4.x) * (step / 6), p.y + (v1.y + v2.y * 2 + v3.y * 2 + v4.y) * (step / 6), p.z + (v1.z + v2.z * 2 + v3.z * 2 + v4.z) * (step / 6)};
                t         = {t.x + (j1.x + j2.x * 2 + j3.x * 2 + j4.x) * (step / 6), t.y + (j1.y + j2.y * 2 + j3.y * 2 + j4.y) * (step / 6), t.z + (j1.z + j2.z * 2 + j3.z * 2 + j4.z) * (step / 6)};
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
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i       = blockIdx.x * tile_cells + local;
            int3 p      = coordinate(layout, i);
            float value = 0;
            if (contains(layout, p)) {
                float3 tangent = map.tangent[axis][i], velocity = sample_velocity(grid, source, map.position[axis][i]);
                value = tangent.x * velocity.x + tangent.y * velocity.y + tangent.z * velocity.z;
                if constexpr (pass == 1) value = .5f * (value - grid.initial.axis[axis][i]);
                if constexpr (pass == 2) {
                    value       = grid.impulse.axis[axis][i] - value;
                    float lower = 1.e30f, upper = -1.e30f;
                    for (int a = 0; a < 3; ++a)
                        for (int sign = -1; sign <= 1; sign += 2) {
                            int3 q{p.x + (a == 0 ? sign : 0), p.y + (a == 1 ? sign : 0), p.z + (a == 2 ? sign : 0)};
                            if (!contains(layout, q)) continue;
                            float neighbor = grid.impulse.axis[axis][index(layout, q)];
                            lower          = fminf(lower, neighbor);
                            upper          = fmaxf(upper, neighbor);
                        }
                    value        = fminf(upper, fmaxf(lower, value));
                    int boundary = grid.boundary[axis][i];
                    if (boundary) value = boundary == 2 ? (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z) : 0;
                }
            }
            destination.axis[axis][i] = value;
        }
    }
    void reconstruct(Grid grid, int steps, float cell_size, float3 wind, float dt, ::cuda::stream_ref stream) {
        float step = dt / cell_size;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), march_kernel<0, true>, grid, steps, step);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), march_kernel<1, true>, grid, steps, step);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), march_kernel<2, true>, grid, steps, step);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), march_kernel<0, false>, grid, steps, step);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), march_kernel<1, false>, grid, steps, step);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), march_kernel<2, false>, grid, steps, step);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<0, 0>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<1, 0>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<2, 0>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<0, 1>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<1, 1>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<2, 1>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<0, 2>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<1, 2>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), pullback_kernel<2, 2>, grid, wind);
    }
}
