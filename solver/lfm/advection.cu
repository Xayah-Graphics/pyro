#include "sampling.cuh"

namespace pyro::cuda::lfm {
    template <int axis>
    __global__ void advect_kernel(Grid grid, Velocity destination, Velocity original, Velocity carrier, float dtOverH, float3 wind) {
        Layout layout = grid.faces[axis];
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i       = blockIdx.x * tileCells + local;
            int3 p      = coordinate(layout, i);
            float value = 0;
            if (contains(layout, p)) {
                int boundary = grid.boundary[axis][i];
                if (boundary) value = boundary == 2 ? component(wind, axis) : 0;
                else {
                    float3 x      = face_position(p, axis);
                    float3 middle = x - sample_velocity(grid, carrier, x) * (.5f * dtOverH);
                    float3 foot   = x - sample_velocity(grid, carrier, middle) * dtOverH;
                    float3 unused;
                    value = sample_axis<false>(layout, original.axis[axis], foot, axis, unused);
                }
            }
            destination.axis[axis][i] = value;
        }
    }
    void advect(Grid grid, Velocity destination, Velocity original, Velocity carrier, float dtOverH, float3 wind, cudaStream_t stream) {
        advect_kernel<0><<<grid.faces[0].blocks, blockThreads, 0, stream>>>(grid, destination, original, carrier, dtOverH, wind);
        advect_kernel<1><<<grid.faces[1].blocks, blockThreads, 0, stream>>>(grid, destination, original, carrier, dtOverH, wind);
        advect_kernel<2><<<grid.faces[2].blocks, blockThreads, 0, stream>>>(grid, destination, original, carrier, dtOverH, wind);
    }
} // namespace pyro::cuda::lfm
