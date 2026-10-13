#include "cuda.cuh"
namespace pyro::cuda::dense {
    __global__ void clear_geometry(Grid grid, float h, int sample) {
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) grid.distance[sample][blockIdx.x * tileCells + local] = 4 * h;
    }
    __global__ void clear_boundary(Grid grid) {
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i                   = blockIdx.x * tileCells + local;
            grid.fluid[i]           = interior(grid.layout, cell_coordinate(grid.layout, i));
            grid.solidDivergence[i] = 0;
            for (int axis = 0; axis < 3; ++axis) {
                grid.aperture[axis][i]      = 1;
                grid.solidVelocity[axis][i] = 0;
            }
        }
    }
    __global__ void geometry_kernel(Grid grid, float h, const Collider* colliders, int count, float time, int sample) {
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        int3 extent = grid.geometryExtent;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        int3 cell{grid.geometryLower.x + ordinal % extent.x, grid.geometryLower.y + ordinal / extent.x % extent.y, grid.geometryLower.z + ordinal / (extent.x * extent.y)};
        int index                    = cell_index(grid.layout, cell);
        grid.distance[sample][index] = fminf(4.0f * h, fmaxf(-4.0f * h, solid_distance(grid.origin + float3{cell.x * h, cell.y * h, cell.z * h}, time, colliders, count)));
    }

    __global__ void boundary_kernel(Grid grid, float h, const Collider* colliders, int count, float time, int sample) {
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        int3 extent = grid.geometryExtent;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        int3 cell{grid.geometryLower.x + ordinal % extent.x, grid.geometryLower.y + ordinal / extent.x % extent.y, grid.geometryLower.z + ordinal / (extent.x * extent.y)};
        int index = cell_index(grid.layout, cell);
        float phi[8], maximum = -1.0e20f;
        for (int k = 0; k < 8; ++k) {
            phi[k]  = value(grid.distance[sample], cell_index(grid.layout, {cell.x + (k & 1), cell.y + ((k >> 1) & 1), cell.z + ((k >> 2) & 1)}), 4.0f * h);
            maximum = fmaxf(maximum, phi[k]);
        }
        grid.fluid[index] = interior(grid.layout, cell) && maximum > 0.0f;
        float volume      = (tetrahedron_fraction(phi[0], phi[1], phi[3], phi[7]) + tetrahedron_fraction(phi[0], phi[3], phi[2], phi[7]) + tetrahedron_fraction(phi[0], phi[2], phi[6], phi[7]) + tetrahedron_fraction(phi[0], phi[6], phi[4], phi[7]) + tetrahedron_fraction(phi[0], phi[4], phi[5], phi[7]) + tetrahedron_fraction(phi[0], phi[5], phi[1], phi[7])) / 6.0f;
        float divergence  = 0;
        if (volume < 1.0f) solid_distance(grid.origin + float3{(cell.x + 0.5f) * h, (cell.y + 0.5f) * h, (cell.z + 0.5f) * h}, time, colliders, count, nullptr, &divergence);
        grid.solidDivergence[index] = (1.0f - volume) * divergence;
        for (int axis = 0; axis < 3; ++axis) {
            int b = 1 << ((axis + 1) % 3), c = 1 << ((axis + 2) % 3);
            grid.aperture[axis][index] = 0.5f * (triangle_fraction(phi[0], phi[b], phi[b | c]) + triangle_fraction(phi[0], phi[b | c], phi[c]));
            float3 p{cell.x + 0.5f, cell.y + 0.5f, cell.z + 0.5f};
            if (axis == 0) p.x -= 0.5f;
            if (axis == 1) p.y -= 0.5f;
            if (axis == 2) p.z -= 0.5f;
            float covered = 1.0f - grid.aperture[axis][index];
            if (covered > 0.0f && covered < 1.0f) {
                float3 y{float(b & 1), float((b >> 1) & 1), float((b >> 2) & 1)};
                float3 z{float(c & 1), float((c >> 1) & 1), float((c >> 2) & 1)};
                float3 moment = (solid_face_moment(phi[0], phi[b], phi[b | c], {}, y, y + z) + solid_face_moment(phi[0], phi[b | c], phi[c], {}, y + z, z)) * 0.5f;
                p             = float3{float(cell.x), float(cell.y), float(cell.z)} + moment / covered;
            }
            float3 velocity{};
            if (covered > 0.0f) solid_distance(grid.origin + p * h, time, colliders, count, &velocity);
            grid.solidVelocity[axis][index] = component(velocity, axis);
        }
    }

    __global__ void initialize_kernel(Grid grid, float3 wind) {
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i = blockIdx.x * tileCells + local;
            Fields fields[]{grid.state, grid.midpoint, grid.next, grid.transported};
            for (auto field : fields) {
                field.smoke[i]       = 0;
                field.temperature[i] = 0;
                for (int axis = 0; axis < 3; ++axis) field.velocity[axis][i] = component(wind, axis);
            }
        }
    }
    void initialize_grid(Grid grid, float3 wind, cudaStream_t stream) {
        initialize_kernel<<<grid.layout.blocks, threads, 0, stream>>>(grid, wind);
    }
    void sample_geometry(Grid grid, float h, const Collider* colliders, int count, float time, int sample, cudaStream_t stream) {
        clear_geometry<<<grid.layout.blocks, threads, 0, stream>>>(grid, h, sample);
        int cells = grid.geometryExtent.x * grid.geometryExtent.y * grid.geometryExtent.z;
        if (cells) geometry_kernel<<<(cells + threads - 1) / threads, threads, 0, stream>>>(grid, h, colliders, count, time, sample);
    }
    void update_boundary(Grid grid, float h, const Collider* colliders, int count, float time, int sample, cudaStream_t stream) {
        clear_boundary<<<grid.layout.blocks, threads, 0, stream>>>(grid);
        int cells = grid.geometryExtent.x * grid.geometryExtent.y * grid.geometryExtent.z;
        if (cells) boundary_kernel<<<(cells + threads - 1) / threads, threads, 0, stream>>>(grid, h, colliders, count, time, sample);
    }
} // namespace pyro::cuda::dense
