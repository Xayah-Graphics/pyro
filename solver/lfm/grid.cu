#include "cuda.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>

namespace pyro::cuda::lfm {
    __global__ void initialize_kernel(Grid grid, float3 wind, int axis) {
        Layout layout = grid.faces[axis];
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i                      = blockIdx.x * tile_cells + local;
            grid.initial.axis[axis][i] = contains(layout, coordinate(layout, i)) ? (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z) : 0;
        }
    }
    __global__ void geometry_kernel(Grid grid, Config config, const Collider* colliders, int count) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i  = blockIdx.x * tile_cells + local;
            int3 p = coordinate(grid.cells, i);
            float3 world{config.origin.x + (p.x + .5f) * config.cell_size, config.origin.y + (p.y + .5f) * config.cell_size, config.origin.z + (p.z + .5f) * config.cell_size};
            grid.fluid[i] = contains(grid.cells, p) && solid_distance(world, 0, colliders, count) >= 0;
        }
    }
    __global__ void boundary_kernel(Grid grid, int axis) {
        Layout layout = grid.faces[axis];
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i    = blockIdx.x * tile_cells + local;
            int3 p   = coordinate(layout, i), previous{p.x + (axis == 0 ? -1 : 0), p.y + (axis == 1 ? -1 : 0), p.z + (axis == 2 ? -1 : 0)};
            int kind = 1;
            if (contains(layout, p)) {
                bool left = contains(grid.cells, previous), right = contains(grid.cells, p);
                bool solid = (left && !grid.fluid[index(grid.cells, previous)]) || (right && !grid.fluid[index(grid.cells, p)]);
                kind       = solid ? 1 : !left || !right ? 2 : 0;
            }
            grid.boundary[axis][i] = static_cast<unsigned char>(kind);
        }
    }
    __global__ void enforce_kernel(Grid grid, Velocity velocity, float3 wind, int axis) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i        = blockIdx.x * tile_cells + local;
            int boundary = grid.boundary[axis][i];
            if (boundary) velocity.axis[axis][i] = boundary == 2 ? (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z) : 0;
        }
    }
    __global__ void measure_kernel(Grid grid, Velocity velocity, float h, double* partial) {
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        double fluid = 0, divergence = 0, speed = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            if (!grid.fluid[i]) continue;
            int3 p      = coordinate(grid.cells, i);
            double flux = 0, v2 = 0;
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
        for (int channel = 0; channel < 3; ++channel) {
            double value = 0;
            for (int i = threadIdx.x; i < blocks; i += blockDim.x) value = channel == 0 ? value + partial[channel * blocks + i] : fmax(value, partial[channel * blocks + i]);
            double result = channel == 0 ? cub::BlockReduce<double, block_threads>(reduction).Sum(value) : cub::BlockReduce<double, block_threads>(reduction).Reduce(value, ::cuda::maximum<>{});
            if (!threadIdx.x) partial[3 * blocks + channel] = result;
            __syncthreads();
        }
    }
    __device__ float centered_velocity(Grid grid, Velocity velocity, int3 p, int axis) {
        p = {min(max(p.x, 0), grid.cells.size.x - 1), min(max(p.y, 0), grid.cells.size.y - 1), min(max(p.z, 0), grid.cells.size.z - 1)};
        return .5f * (velocity.axis[axis][index(grid.faces[axis], p)] + velocity.axis[axis][index(grid.faces[axis], int3{p.x + (axis == 0 ? 1 : 0), p.y + (axis == 1 ? 1 : 0), p.z + (axis == 2 ? 1 : 0)})]);
    }
    __global__ void vorticity_kernel(Grid grid, Velocity velocity, float h) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i  = blockIdx.x * tile_cells + local;
            int3 p = coordinate(grid.cells, i);
            if (!contains(grid.cells, p)) continue;
            float3 curl{};
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
    void initialize(Grid grid, Config config, cudaStream_t stream) {
        for (int axis = 0; axis < 3; ++axis) initialize_kernel<<<grid.faces[axis].blocks, block_threads, 0, stream>>>(grid, config.wind, axis);
    }
    void geometry(Grid grid, Config config, const Collider* colliders, int count, cudaStream_t stream) {
        geometry_kernel<<<grid.cells.blocks, block_threads, 0, stream>>>(grid, config, colliders, count);
        for (int axis = 0; axis < 3; ++axis) boundary_kernel<<<grid.faces[axis].blocks, block_threads, 0, stream>>>(grid, axis);
    }
    void enforce_boundary(Grid grid, Velocity velocity, float3 wind, cudaStream_t stream) {
        for (int axis = 0; axis < 3; ++axis) enforce_kernel<<<grid.faces[axis].blocks, block_threads, 0, stream>>>(grid, velocity, wind, axis);
    }
    void measure(Grid grid, Velocity velocity, float h, double* reductions, cudaStream_t stream) {
        measure_kernel<<<grid.cells.blocks, block_threads, 0, stream>>>(grid, velocity, h, reductions);
        finish_measure<<<1, block_threads, 0, stream>>>(reductions, grid.cells.blocks);
    }
    void export_vorticity(Grid grid, Velocity velocity, float h, cudaStream_t stream) {
        vorticity_kernel<<<grid.cells.blocks, block_threads, 0, stream>>>(grid, velocity, h);
    }
} // namespace pyro::cuda::lfm
