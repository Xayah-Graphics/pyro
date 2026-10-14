#include "cuda.cuh"

namespace pyro::cuda::sparse {
    __global__ void remap_scalars(Grid oldGrid, Grid grid) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        float3 p    = center(grid.layout, i);
        float h     = spacing(grid.layout, i);
        int old     = locate(oldGrid.layout, p);
        float smoke = 0, temperature = 0;
        if (old >= 0) {
            float oldH = spacing(oldGrid.layout, old);
            if (oldH == h) {
                smoke       = oldGrid.state.smoke[old];
                temperature = oldGrid.state.temperature[old];
            } else if (oldH > h) {
                smoke       = oldGrid.state.smoke[old] + dot(scalar_gradient(oldGrid, oldGrid.state.smoke, old), p - center(oldGrid.layout, old));
                temperature = oldGrid.state.temperature[old] + dot(scalar_gradient(oldGrid, oldGrid.state.temperature, old), p - center(oldGrid.layout, old));
            } else {
                int n      = 1 << grid.layout.blocks[i >> 9].level;
                double sum = 0, heat = 0;
                for (int z = 0; z < n; ++z)
                    for (int y = 0; y < n; ++y)
                        for (int x = 0; x < n; ++x) {
                            float3 q = p + float3{(x + .5f) * grid.layout.h - .5f * h, (y + .5f) * grid.layout.h - .5f * h, (z + .5f) * grid.layout.h - .5f * h};
                            int j    = locate(oldGrid.layout, q);
                            if (j >= 0) {
                                sum += oldGrid.state.smoke[j];
                                heat += oldGrid.state.temperature[j];
                            }
                        }
                smoke       = float(sum / (n * n * n));
                temperature = float(heat / (n * n * n));
            }
        }
        grid.state.smoke[i]       = smoke;
        grid.state.temperature[i] = temperature;
    }
    __device__ float old_flux(Grid grid, float3 p, int axis, float wind) {
        float epsilon = grid.layout.h * .001f;
        int left = locate(grid.layout, p - axis_vector(axis, epsilon)), right = locate(grid.layout, p + axis_vector(axis, epsilon));
        int i = right;
        if (left >= 0 && (right < 0 || spacing(grid.layout, left) < spacing(grid.layout, right))) i = left;
        if (i < 0) return wind;
        float h = spacing(grid.layout, i), t = .5f + component(p - center(grid.layout, i), axis) / h;
        float l = side_velocity(grid, grid.state, i, axis, -1, true), r = side_velocity(grid, grid.state, i, axis, 1, true);
        return l + (r - l) * t;
    }
    __global__ void remap_faces(Grid oldGrid, Grid grid, Config config) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.left < 0 && face.right < 0) {
            grid.state.velocity[f] = 0;
            return;
        }
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        int previous = find_block(oldGrid.layout, grid.layout.blocks[owner >> 9].coordinate, grid.layout.blocks[owner >> 9].level);
        if (previous >= 0) {
            int oldFace = previous * 1728 + f % 1728;
            if (oldGrid.faces[oldFace].left >= 0 || oldGrid.faces[oldFace].right >= 0) {
                grid.state.velocity[f] = oldGrid.state.velocity[oldFace];
                return;
            }
        }
        int n       = 1 << grid.layout.blocks[owner >> 9].level;
        double flux = 0;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                float3 q = p + axis_vector((axis + 1) % 3, (x + .5f) * grid.layout.h - h * .5f) + axis_vector((axis + 2) % 3, (y + .5f) * grid.layout.h - h * .5f);
                flux += old_flux(oldGrid, q, axis, component(config.wind, axis));
            }
        grid.state.velocity[f] = face.area > 0 ? float((flux * (h * h / (n * n)) - face.solidFlux) / face.area) : face.solidFlux / (h * h);
    }
    __global__ void preserve_child_divergence(Grid oldGrid, Grid grid) {
        int old = blockIdx.x * blockDim.x + threadIdx.x;
        if (old >= oldGrid.layout.count * 512 || !oldGrid.fluid[old]) return;
        float h = spacing(oldGrid.layout, old);
        if (h == grid.layout.h) return;
        float3 c = center(oldGrid.layout, old);
        int children[8];
        for (int k = 0; k < 8; ++k) {
            children[k] = locate(grid.layout, c + float3{(k & 1 ? .25f : -.25f) * h, (k & 2 ? .25f : -.25f) * h, (k & 4 ? .25f : -.25f) * h});
            if (children[k] < 0 || spacing(grid.layout, children[k]) != h * .5f || !grid.fluid[children[k]]) return;
        }
        float rhs[8], potential[8]{};
        float parentFlux = 0;
        for (int a = 0; a < 3; ++a) parentFlux += (side_velocity(oldGrid, oldGrid.state, old, a, 1, true) - side_velocity(oldGrid, oldGrid.state, old, a, -1, true)) * h * h;
        for (int k = 0; k < 8; ++k) {
            float flux = 0;
            for (int a = 0; a < 3; ++a) flux += (side_velocity(grid, grid.state, children[k], a, 1, true) - side_velocity(grid, grid.state, children[k], a, -1, true)) * h * h * .25f;
            rhs[k] = flux - parentFlux * .125f;
        }
        // Exact Neumann solve on the eight-child cube using its seven nonconstant
        // Walsh eigenmodes. Exterior fluxes remain untouched.
        for (int mode = 1; mode < 8; ++mode) {
            float coefficient = 0;
            for (int k = 0; k < 8; ++k) coefficient += (__popc(unsigned(k & mode)) & 1 ? -1 : 1) * rhs[k];
            coefficient /= 16.f * __popc(unsigned(mode));
            for (int k = 0; k < 8; ++k) potential[k] += (__popc(unsigned(k & mode)) & 1 ? -1 : 1) * coefficient;
        }
        for (int k = 0; k < 8; ++k)
            for (int a = 0; a < 3; ++a)
                if (!(k & (1 << a))) {
                    int f = negative_face(children[k | (1 << a)], a);
                    if (grid.faces[f].area > 0) grid.state.velocity[f] -= (potential[k] - potential[k | (1 << a)]) / grid.faces[f].area;
                }
    }
    void remap(Grid oldGrid, Grid newGrid, Config config, cudaStream_t stream) {
        if (newGrid.layout.count) {
            remap_scalars<<<newGrid.layout.count * 2, 256, 0, stream>>>(oldGrid, newGrid);
            remap_faces<<<(newGrid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(oldGrid, newGrid, config);
            if (oldGrid.layout.count) preserve_child_divergence<<<oldGrid.layout.count * 2, 256, 0, stream>>>(oldGrid, newGrid);
        }
    }
} // namespace pyro::cuda::sparse
