#include "advection.cuh"
#include <cuda/launch>

namespace pyro::lfm {
    template <int axis>
    __global__ void advect_kernel(Grid grid, Velocity destination, Velocity original, Velocity carrier, float dt_over_h, float3 wind) {
        Layout layout = grid.faces[axis];
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i       = blockIdx.x * tile_cells + local;
            int3 p      = coordinate(layout, i);
            float value = 0;
            if (contains(layout, p)) {
                int boundary = grid.boundary[axis][i];
                if (boundary) value = boundary == 2 ? (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z) : 0;
                else {
                    float3 x     = face_position(p, axis);
                    float3 first = sample_velocity(grid, carrier, x);
                    float3 middle{x.x - first.x * (.5f * dt_over_h), x.y - first.y * (.5f * dt_over_h), x.z - first.z * (.5f * dt_over_h)};
                    float3 second = sample_velocity(grid, carrier, middle);
                    float3 foot{x.x - second.x * dt_over_h, x.y - second.y * dt_over_h, x.z - second.z * dt_over_h};
                    float3 unused;
                    value = sample_axis<false>(layout, original.axis[axis], foot, axis, unused);
                }
            }
            destination.axis[axis][i] = value;
        }
    }

    void advect(Grid grid, Velocity destination, Velocity original, Velocity carrier, float dt_over_h, float3 wind, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), advect_kernel<0>, grid, destination, original, carrier, dt_over_h, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), advect_kernel<1>, grid, destination, original, carrier, dt_over_h, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), advect_kernel<2>, grid, destination, original, carrier, dt_over_h, wind);
    }
}
