#include "statistics.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cuda/launch>

namespace pyro::lfm {
    __device__ float centered_velocity(Grid grid, Velocity velocity, int3 p, int axis) {
        p = {min(max(p.x, 0), grid.cells.size.x - 1), min(max(p.y, 0), grid.cells.size.y - 1), min(max(p.z, 0), grid.cells.size.z - 1)};
        return .5f * (velocity.axis[axis][index(grid.faces[axis], p)] + velocity.axis[axis][index(grid.faces[axis], int3{p.x + (axis == 0 ? 1 : 0), p.y + (axis == 1 ? 1 : 0), p.z + (axis == 2 ? 1 : 0)})]);
    }

    __global__ void measure_kernel(Grid grid, Velocity velocity, float h, double* partial) {
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        double fluid = 0, divergence = 0, speed = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            if (!grid.fluid[i]) continue;
            int3 p      = coordinate(grid.cells, i);
            double flux = 0, v2 = 0;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                double a = velocity.axis[axis][index(grid.faces[axis], p)];
                double b = velocity.axis[axis][index(grid.faces[axis], int3{p.x + (axis == 0 ? 1 : 0), p.y + (axis == 1 ? 1 : 0), p.z + (axis == 2 ? 1 : 0)})];
                flux += b - a;
                double largest = fmax(fabs(a), fabs(b));
                v2 += largest * largest;
            }
            fluid += 1;
            divergence = fmax(divergence, fabs(flux) / h);
            speed      = fmax(speed, sqrt(v2));
        }
        double value = cub::BlockReduce<double, block_threads>(reduction).Sum(fluid);
        if (!threadIdx.x) partial[blockIdx.x] = value;
        __syncthreads();
        value = cub::BlockReduce<double, block_threads>(reduction).Reduce(divergence, ::cuda::maximum<>{});
        if (!threadIdx.x) partial[grid.cells.blocks + blockIdx.x] = value;
        __syncthreads();
        value = cub::BlockReduce<double, block_threads>(reduction).Reduce(speed, ::cuda::maximum<>{});
        if (!threadIdx.x) partial[2 * grid.cells.blocks + blockIdx.x] = value;
    }

    __global__ void finish_measure(double* partial, int blocks) {
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
#pragma unroll
        for (int channel = 0; channel < 3; ++channel) {
            double value = 0;
            for (int i = threadIdx.x; i < blocks; i += blockDim.x) value = channel == 0 ? value + partial[channel * blocks + i] : fmax(value, partial[channel * blocks + i]);
            double result = channel == 0 ? cub::BlockReduce<double, block_threads>(reduction).Sum(value) : cub::BlockReduce<double, block_threads>(reduction).Reduce(value, ::cuda::maximum<>{});
            if (!threadIdx.x) partial[3 * blocks + channel] = result;
            __syncthreads();
        }
    }

    __global__ void vorticity_kernel(Grid grid, Velocity velocity, float h) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i  = blockIdx.x * tile_cells + local;
            int3 p = coordinate(grid.cells, i);
            if (!contains(grid.cells, p)) continue;
            float3 curl{};
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                int b = (axis + 1) % 3, c = (axis + 2) % 3;
                float v = (centered_velocity(grid, velocity, int3{p.x + (b == 0 ? 1 : 0), p.y + (b == 1 ? 1 : 0), p.z + (b == 2 ? 1 : 0)}, c) - centered_velocity(grid, velocity, int3{p.x + (b == 0 ? -1 : 0), p.y + (b == 1 ? -1 : 0), p.z + (b == 2 ? -1 : 0)}, c) - centered_velocity(grid, velocity, int3{p.x + (c == 0 ? 1 : 0), p.y + (c == 1 ? 1 : 0), p.z + (c == 2 ? 1 : 0)}, b) + centered_velocity(grid, velocity, int3{p.x + (c == 0 ? -1 : 0), p.y + (c == 1 ? -1 : 0), p.z + (c == 2 ? -1 : 0)}, b)) * (.5f / h);
                if (axis == 0) curl.x = v;
                if (axis == 1) curl.y = v;
                if (axis == 2) curl.z = v;
            }
            int output          = (p.z * grid.cells.size.y + p.y) * grid.cells.size.x + p.x;
            grid.output[output] = grid.fluid[i] ? sqrtf(curl.x * curl.x + curl.y * curl.y + curl.z * curl.z) : 0;
        }
    }

    void measure(Grid grid, Velocity velocity, float h, double* reductions, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.cells.blocks), ::cuda::block_dims<block_threads>()), measure_kernel, grid, velocity, h, reductions);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<block_threads>()), finish_measure, reductions, grid.cells.blocks);
    }

    void export_vorticity(Grid grid, Velocity velocity, float h, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.cells.blocks), ::cuda::block_dims<block_threads>()), vorticity_kernel, grid, velocity, h);
    }

}
