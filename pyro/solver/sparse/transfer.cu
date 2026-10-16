#include "transfer.cuh"

#include <cuda/launch>

namespace pyro::sparse {
    __device__ float old_flux(Grid grid, float3 p, int axis, float wind) {
        float epsilon = grid.layout.h * .001f;
        int left = locate(grid.layout, float3{p.x - (axis == 0 ? epsilon : 0), p.y - (axis == 1 ? epsilon : 0), p.z - (axis == 2 ? epsilon : 0)}), right = locate(grid.layout, float3{p.x + (axis == 0 ? epsilon : 0), p.y + (axis == 1 ? epsilon : 0), p.z + (axis == 2 ? epsilon : 0)});
        int i = right;
        if (left >= 0 && (right < 0 || spacing(grid.layout, left) < spacing(grid.layout, right))) i = left;
        if (i < 0) return wind;
        float3 c = center(grid.layout, i);
        float h = spacing(grid.layout, i), t = .5f + (axis == 0 ? p.x - c.x : axis == 1 ? p.y - c.y : p.z - c.z) / h;
        float l = side_velocity(grid, grid.state, i, axis, -1, true), r = side_velocity(grid, grid.state, i, axis, 1, true);
        return l + (r - l) * t;
    }

    template <bool previous>
    __global__ void remap_scalars(Grid old_grid, Grid grid) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        float3 p    = center(grid.layout, i);
        float h     = spacing(grid.layout, i);
        int old     = previous ? locate(old_grid.layout, p) : -1;
        float smoke = 0, temperature = 0;
        if (old >= 0) {
            float old_h = spacing(old_grid.layout, old);
            if (old_h == h) {
                smoke       = old_grid.state.smoke[old];
                temperature = old_grid.state.temperature[old];
            } else if (old_h > h) {
                float3 c = center(old_grid.layout, old), smoke_gradient = scalar_gradient(old_grid, old_grid.state.smoke, old), temperature_gradient = scalar_gradient(old_grid, old_grid.state.temperature, old);
                float3 delta{p.x - c.x, p.y - c.y, p.z - c.z};
                smoke       = old_grid.state.smoke[old] + (smoke_gradient.x * delta.x + smoke_gradient.y * delta.y + smoke_gradient.z * delta.z);
                temperature = old_grid.state.temperature[old] + (temperature_gradient.x * delta.x + temperature_gradient.y * delta.y + temperature_gradient.z * delta.z);
            } else {
                int n      = 1 << grid.layout.blocks[i >> 9].level;
                double sum = 0, heat = 0;
                for (int z = 0; z < n; ++z)
                    for (int y = 0; y < n; ++y)
                        for (int x = 0; x < n; ++x) {
                            float3 q{p.x + ((x + .5f) * grid.layout.h - .5f * h), p.y + ((y + .5f) * grid.layout.h - .5f * h), p.z + ((z + .5f) * grid.layout.h - .5f * h)};
                            int j = locate(old_grid.layout, q);
                            if (j >= 0) {
                                sum += old_grid.state.smoke[j];
                                heat += old_grid.state.temperature[j];
                            }
                        }
                smoke       = float(sum / (n * n * n));
                temperature = float(heat / (n * n * n));
            }
        }
        grid.state.smoke[i]       = smoke;
        grid.state.temperature[i] = temperature;
    }

    template <bool previous>
    __global__ void remap_faces(Grid old_grid, Grid grid, float3 wind) {
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
        if constexpr (!previous) {
            double flux            = double(axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z) * (h * h);
            grid.state.velocity[f] = face.area > 0 ? float((flux - face.solid_flux) / face.area) : face.solid_flux / (h * h);
            return;
        }
        int old_block = find_block(old_grid.layout, grid.layout.blocks[owner >> 9].coordinate, grid.layout.blocks[owner >> 9].level);
        if (old_block >= 0) {
            int old_face = old_block * 1728 + f % 1728;
            if (old_grid.faces[old_face].left >= 0 || old_grid.faces[old_face].right >= 0) {
                grid.state.velocity[f] = old_grid.state.velocity[old_face];
                return;
            }
        }
        int n       = 1 << grid.layout.blocks[owner >> 9].level;
        double flux = 0;
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                int a = (axis + 1) % 3, b = (axis + 2) % 3;
                float u = (x + .5f) * grid.layout.h - h * .5f, v = (y + .5f) * grid.layout.h - h * .5f;
                float3 q{p.x + (a == 0 ? u : 0) + (b == 0 ? v : 0), p.y + (a == 1 ? u : 0) + (b == 1 ? v : 0), p.z + (a == 2 ? u : 0) + (b == 2 ? v : 0)};
                flux += old_flux(old_grid, q, axis, (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z));
            }
        grid.state.velocity[f] = face.area > 0 ? float((flux * (h * h / (n * n)) - face.solid_flux) / face.area) : face.solid_flux / (h * h);
    }

    __global__ void preserve_child_divergence(Grid old_grid, Grid grid) {
        int old = blockIdx.x * blockDim.x + threadIdx.x;
        if (old >= old_grid.layout.count * 512 || !old_grid.fluid[old]) return;
        float h = spacing(old_grid.layout, old);
        if (h == grid.layout.h) return;
        float3 c = center(old_grid.layout, old);
        int children[8];
        for (int k = 0; k < 8; ++k) {
            children[k] = locate(grid.layout, float3{c.x + (k & 1 ? .25f : -.25f) * h, c.y + (k & 2 ? .25f : -.25f) * h, c.z + (k & 4 ? .25f : -.25f) * h});
            if (children[k] < 0 || spacing(grid.layout, children[k]) != h * .5f || !grid.fluid[children[k]]) return;
        }
        float rhs[8], potential[8]{};
        float parent_flux = 0;
        for (int a = 0; a < 3; ++a) parent_flux += (side_velocity(old_grid, old_grid.state, old, a, 1, true) - side_velocity(old_grid, old_grid.state, old, a, -1, true)) * h * h;
        for (int k = 0; k < 8; ++k) {
            float flux = 0;
            for (int a = 0; a < 3; ++a) flux += (side_velocity(grid, grid.state, children[k], a, 1, true) - side_velocity(grid, grid.state, children[k], a, -1, true)) * h * h * .25f;
            rhs[k] = flux - parent_flux * .125f;
        }
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

    void remap(Grid old_grid, Grid new_grid, float3 wind, ::cuda::stream_ref stream) {
        if (new_grid.layout.count) {
            if (old_grid.layout.count) {
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(new_grid.layout.count * 2), ::cuda::block_dims<256>()), remap_scalars<true>, old_grid, new_grid);
                ::cuda::launch(stream, ::cuda::distribute<256>(new_grid.layout.count * 1728), remap_faces<true>, old_grid, new_grid, wind);
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(old_grid.layout.count * 2), ::cuda::block_dims<256>()), preserve_child_divergence, old_grid, new_grid);
            } else {
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(new_grid.layout.count * 2), ::cuda::block_dims<256>()), remap_scalars<false>, old_grid, new_grid);
                ::cuda::launch(stream, ::cuda::distribute<256>(new_grid.layout.count * 1728), remap_faces<false>, old_grid, new_grid, wind);
            }
        }
    }

}
