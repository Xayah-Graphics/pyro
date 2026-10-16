#include "grid.cuh"
#include <cuda/launch>
#include <solver/common/scene.cuh>

namespace pyro::lfm {
    template <int axis>
    __global__ void initialize_kernel(Grid grid, float3 wind) {
        Layout layout = grid.faces[axis];
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i                      = blockIdx.x * tile_cells + local;
            grid.initial.axis[axis][i] = contains(layout, coordinate(layout, i)) ? (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z) : 0;
        }
    }

    template <bool obstacles>
    __global__ void geometry_kernel(Grid grid, float cell_size, float3 origin, const ColliderView* colliders, int count) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i  = blockIdx.x * tile_cells + local;
            int3 p = coordinate(grid.cells, i);
            float3 world{origin.x + (p.x + .5f) * cell_size, origin.y + (p.y + .5f) * cell_size, origin.z + (p.z + .5f) * cell_size};
            grid.fluid[i] = contains(grid.cells, p) && (!obstacles || solid_distance(world, 0, colliders, count, 4 * cell_size) >= 0);
        }
    }

    template <int axis>
    __global__ void boundary_kernel(Grid grid) {
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

    template <int axis>
    __global__ void enforce_kernel(Grid grid, Velocity velocity, float3 wind) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i        = blockIdx.x * tile_cells + local;
            int boundary = grid.boundary[axis][i];
            if (boundary) velocity.axis[axis][i] = boundary == 2 ? (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z) : 0;
        }
    }

    void initialize(Grid grid, float3 wind, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), initialize_kernel<0>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), initialize_kernel<1>, grid, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), initialize_kernel<2>, grid, wind);
    }

    void update_geometry(Grid grid, float cell_size, float3 origin, const ColliderView* colliders, int count, ::cuda::stream_ref stream) {
        if (count) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.cells.blocks), ::cuda::block_dims<block_threads>()), geometry_kernel<true>, grid, cell_size, origin, colliders, count);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.cells.blocks), ::cuda::block_dims<block_threads>()), geometry_kernel<false>, grid, cell_size, origin, colliders, count);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), boundary_kernel<0>, grid);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), boundary_kernel<1>, grid);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), boundary_kernel<2>, grid);
    }

    void enforce_boundary(Grid grid, Velocity velocity, float3 wind, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), enforce_kernel<0>, grid, velocity, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), enforce_kernel<1>, grid, velocity, wind);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), enforce_kernel<2>, grid, velocity, wind);
    }
}
