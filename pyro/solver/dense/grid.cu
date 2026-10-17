#include "grid.cuh"

#include <cuda/launch>

#include <geometry/clipping.cuh>
#include <solver/common/scene.cuh>

namespace pyro::dense {
    __global__ void initialize_kernel(Grid grid, float3 wind) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            Fields fields[]{grid.state, grid.midpoint, grid.next};
            for (auto field : fields) {
                field.smoke[i]       = 0;
                field.temperature[i] = 0;
                for (int axis = 0; axis < 3; ++axis) field.velocity[axis][i] = (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z);
            }
            for (int axis = 0; axis < 3; ++axis) grid.transported.velocity[axis][i] = (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z);
        }
    }

    __global__ void clear_geometry(Grid grid, float h, float* distance) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) distance[blockIdx.x * tile_cells + local] = 4 * h;
    }

    __global__ void clear_boundary(Grid grid) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i                    = blockIdx.x * tile_cells + local;
            grid.fluid[i]            = interior(grid.layout, cell_coordinate(grid.layout, i));
            grid.solid_divergence[i] = 0;
            for (int axis = 0; axis < 3; ++axis) {
                grid.aperture[axis][i]       = 1;
                grid.solid_velocity[axis][i] = 0;
            }
        }
    }

    __global__ void geometry_kernel(Grid grid, float h, const ColliderView* colliders, int count, float time, float* distance) {
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        int3 extent = grid.geometry_extent;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        int3 cell{grid.geometry_lower.x + ordinal % extent.x, grid.geometry_lower.y + ordinal / extent.x % extent.y, grid.geometry_lower.z + ordinal / (extent.x * extent.y)};
        int index       = cell_index(grid.layout, cell);
        distance[index] = fminf(4.0f * h, fmaxf(-4.0f * h, solid_distance({grid.origin.x + cell.x * h, grid.origin.y + cell.y * h, grid.origin.z + cell.z * h}, time, colliders, count, 4 * h)));
    }

    template <bool moving>
    __global__ void boundary_kernel(Grid grid, float h, const ColliderView* colliders, int count, float time, const float* distance) {
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        int3 extent = grid.geometry_extent;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        int3 cell{grid.geometry_lower.x + ordinal % extent.x, grid.geometry_lower.y + ordinal / extent.x % extent.y, grid.geometry_lower.z + ordinal / (extent.x * extent.y)};
        int index = cell_index(grid.layout, cell);
        float phi[8], maximum = -1.0e20f;
        for (int k = 0; k < 8; ++k) {
            phi[k]  = value(distance, cell_index(grid.layout, {cell.x + (k & 1), cell.y + ((k >> 1) & 1), cell.z + ((k >> 2) & 1)}), 4.0f * h);
            maximum = fmaxf(maximum, phi[k]);
        }
        grid.fluid[index]            = interior(grid.layout, cell) && maximum > 0.0f;
        grid.solid_divergence[index] = 0;
        if constexpr (moving) {
            float volume     = (::geometry::tetrahedron_positive_fraction(phi[0], phi[1], phi[3], phi[7]) + ::geometry::tetrahedron_positive_fraction(phi[0], phi[3], phi[2], phi[7]) + ::geometry::tetrahedron_positive_fraction(phi[0], phi[2], phi[6], phi[7]) + ::geometry::tetrahedron_positive_fraction(phi[0], phi[6], phi[4], phi[7]) + ::geometry::tetrahedron_positive_fraction(phi[0], phi[4], phi[5], phi[7]) + ::geometry::tetrahedron_positive_fraction(phi[0], phi[5], phi[1], phi[7])) / 6.0f;
            float divergence = 0;
            if (volume < 1.0f) solid_distance<false, true>({grid.origin.x + (cell.x + 0.5f) * h, grid.origin.y + (cell.y + 0.5f) * h, grid.origin.z + (cell.z + 0.5f) * h}, time, colliders, count, 4 * h, nullptr, &divergence);
            grid.solid_divergence[index] = (1.0f - volume) * divergence;
        }
        for (int axis = 0; axis < 3; ++axis) {
            int b = 1 << ((axis + 1) % 3), c = 1 << ((axis + 2) % 3);
            grid.aperture[axis][index]       = 0.5f * (::geometry::triangle_positive_fraction(phi[0], phi[b], phi[b | c]) + ::geometry::triangle_positive_fraction(phi[0], phi[b | c], phi[c]));
            grid.solid_velocity[axis][index] = 0;
            if constexpr (moving) {
                float3 p{cell.x + 0.5f, cell.y + 0.5f, cell.z + 0.5f};
                if (axis == 0) p.x -= 0.5f;
                if (axis == 1) p.y -= 0.5f;
                if (axis == 2) p.z -= 0.5f;
                float covered = 1.0f - grid.aperture[axis][index];
                if (covered > 0.0f && covered < 1.0f) {
                    float3 y{float(b & 1), float((b >> 1) & 1), float((b >> 2) & 1)};
                    float3 z{float(c & 1), float((c >> 1) & 1), float((c >> 2) & 1)};
                    float3 corner{y.x + z.x, y.y + z.y, y.z + z.z};
                    float3 first = ::geometry::triangle_negative_moment(phi[0], phi[b], phi[b | c], {}, y, corner), second = ::geometry::triangle_negative_moment(phi[0], phi[b | c], phi[c], {}, corner, z);
                    float3 moment{(first.x + second.x) * 0.5f, (first.y + second.y) * 0.5f, (first.z + second.z) * 0.5f};
                    float inverse_covered = 1.0f / covered;
                    p                     = {float(cell.x) + moment.x * inverse_covered, float(cell.y) + moment.y * inverse_covered, float(cell.z) + moment.z * inverse_covered};
                }
                float3 velocity{};
                if (covered > 0.0f) solid_distance<true>({grid.origin.x + p.x * h, grid.origin.y + p.y * h, grid.origin.z + p.z * h}, time, colliders, count, 4 * h, &velocity);
                grid.solid_velocity[axis][index] = (axis == 0 ? velocity.x : axis == 1 ? velocity.y : velocity.z);
            }
        }
    }

    void initialize_grid(Grid grid, float3 wind, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), initialize_kernel, grid, wind);
    }

    void sample_geometry(Grid grid, float h, const ColliderView* colliders, int count, float time, int sample, ::cuda::stream_ref stream) {
        if (grid.boundary_mode == BoundaryMode::none) return;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), clear_geometry, grid, h, grid.distance[sample]);
        int cells = grid.geometry_extent.x * grid.geometry_extent.y * grid.geometry_extent.z;
        if (cells) ::cuda::launch(stream, ::cuda::distribute<threads>(cells), geometry_kernel, grid, h, colliders, count, time, grid.distance[sample]);
    }

    void update_boundary(Grid grid, float h, const ColliderView* colliders, int count, float time, int sample, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), clear_boundary, grid);
        int cells = grid.geometry_extent.x * grid.geometry_extent.y * grid.geometry_extent.z;
        if (cells) {
            if (grid.boundary_mode == BoundaryMode::moving) ::cuda::launch(stream, ::cuda::distribute<threads>(cells), boundary_kernel<true>, grid, h, colliders, count, time, grid.distance[sample]);
            else ::cuda::launch(stream, ::cuda::distribute<threads>(cells), boundary_kernel<false>, grid, h, colliders, count, time, grid.distance[sample]);
        }
    }

}
