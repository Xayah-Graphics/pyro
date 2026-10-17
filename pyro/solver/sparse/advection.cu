#include "advection.cuh"
#include <cuda/launch>

namespace pyro::sparse {
    template <BoundaryMode boundary_mode>
    __device__ float3 trace(Grid grid, Fields carrier, float cell_size, float3 wind, float3 p, float time, float dt, bool& clipped) {
        clipped = false;
        float h = cell_size;
        if constexpr (boundary_mode != BoundaryMode::none) {
            float distance = sample_boundary<boundary_mode>(grid, p, time);
            if (distance < .01f * h) {
                float3 gradient{};
                float dx    = sample_boundary<boundary_mode>(grid, {p.x + h * .25f, p.y, p.z}, time) - sample_boundary<boundary_mode>(grid, {p.x - h * .25f, p.y, p.z}, time);
                float dy    = sample_boundary<boundary_mode>(grid, {p.x, p.y + h * .25f, p.z}, time) - sample_boundary<boundary_mode>(grid, {p.x, p.y - h * .25f, p.z}, time);
                float dz    = sample_boundary<boundary_mode>(grid, {p.x, p.y, p.z + h * .25f}, time) - sample_boundary<boundary_mode>(grid, {p.x, p.y, p.z - h * .25f}, time);
                gradient    = {dx, dy, dz};
                float scale = (.01f * h - distance) / fmaxf(sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z), 1.e-20f);
                p           = {p.x + gradient.x * scale, p.y + gradient.y * scale, p.z + gradient.z * scale};
                clipped     = true;
            }
        }
        float3 v = sample_velocity(grid, carrier, p, wind), middle = sample_velocity(grid, carrier, {p.x - v.x * (.5f * dt), p.y - v.y * (.5f * dt), p.z - v.z * (.5f * dt)}, wind);
        float3 departure{p.x - middle.x * dt, p.y - middle.y * dt, p.z - middle.z * dt};
        if constexpr (boundary_mode == BoundaryMode::none) return departure;
        float3 delta{departure.x - p.x, departure.y - p.y, departure.z - p.z};
        float bound = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) + (boundary_mode == BoundaryMode::moving ? grid.collider_speed * fabsf(dt) : 0), fraction = 0;
        if (bound < 1.e-12f) return departure;
        for (int step = 0; step < 64; ++step) {
            float3 q{p.x + delta.x * fraction, p.y + delta.y * fraction, p.z + delta.z * fraction};
            float d = sample_boundary<boundary_mode>(grid, q, time - dt * fraction);
            if (d < .001f * h) {
                clipped = true;
                return q;
            }
            fraction += .5f * d / bound;
            if (fraction >= 1) return departure;
        }
        clipped = true;
        return {p.x + delta.x * fraction, p.y + delta.y * fraction, p.z + delta.z * fraction};
    }

    template <bool correct, BoundaryMode boundary_mode>
    __global__ void transport_cells(Grid grid, Fields destination, Fields carrier, float cell_size, float3 wind, float time, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        if constexpr (boundary_mode != BoundaryMode::none)
            if (!grid.fluid[i]) {
                destination.smoke[i] = destination.temperature[i] = 0;
                return;
            }
        float3 p = center(grid.layout, i);
        bool clipped, reverse_clipped = true;
        float3 departure = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time + dt, dt, clipped), reverse{};
        if (correct && !clipped) reverse = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time, -dt, reverse_clipped);
        int departure_anchor = locate(grid.layout, departure);
        int reverse_anchor = correct && !clipped && !reverse_clipped ? locate(grid.layout, reverse) : -1;
#pragma unroll
        for (int field = 3; field < 5; ++field) {
            float lo, hi, value = sample<correct>(grid, grid.state, departure, field, departure_anchor, 0, &lo, &hi);
            if (correct && !clipped && !reverse_clipped) value = fminf(hi, fmaxf(lo, value + .5f * ((field == 3 ? grid.state.smoke[i] : grid.state.temperature[i]) - sample(grid, grid.transported, reverse, field, reverse_anchor))));
            (field == 3 ? destination.smoke : destination.temperature)[i] = value;
        }
    }

    template <bool correct, BoundaryMode boundary_mode>
    __global__ void transport_faces(Grid grid, Fields destination, Fields carrier, float cell_size, float3 wind, float time, float dt) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.left < 0 && face.right < 0) {
            destination.velocity[f] = 0;
            return;
        }
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        if constexpr (boundary_mode != BoundaryMode::none)
            if (face.area <= 0) {
                destination.velocity[f] = face.solid_flux / (h * h);
                return;
            }
        bool clipped;
        float3 departure = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time + dt, dt, clipped);
        float lo, hi, value = sample<correct>(grid, grid.state, departure, axis, locate(grid.layout, departure), (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z), &lo, &hi);
        if (correct && !clipped) {
            float3 reverse = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time, -dt, clipped);
            if (!clipped) value = fminf(hi, fmaxf(lo, value + .5f * (grid.state.velocity[f] - sample(grid, grid.transported, reverse, axis, locate(grid.layout, reverse), (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z)))));
        }
        destination.velocity[f] = value;
    }

    __global__ void cache_samples(Grid grid, Fields fields, float3 wind, float* output) {
        int block = blockIdx.x, field = blockIdx.y;
        Block node = grid.layout.blocks[block];
        int level  = node.level;
        float h    = ldexpf(grid.layout.h, level);
        float3 stagger{field == 0 ? 0.0f : .5f, field == 1 ? 0.0f : .5f, field == 2 ? 0.0f : .5f};
        float background = field < 3 ? (field == 0 ? wind.x : field == 1 ? wind.y : wind.z) : 0;
        for (int local = threadIdx.x; local < 1331; local += blockDim.x) {
            int3 c{node.coordinate.x * 8 + local % 11 - 1, node.coordinate.y * 8 + (local / 11) % 11 - 1, node.coordinate.z * 8 + local / 121 - 1};
            int b        = (c.x >> 3) == node.coordinate.x && (c.y >> 3) == node.coordinate.y && (c.z >> 3) == node.coordinate.z ? block : find_block(grid.layout, {c.x >> 3, c.y >> 3, c.z >> 3}, level);
            float value  = background;
            bool uniform = b >= 0;
            float3 point{(c.x + stagger.x) * h, (c.y + stagger.y) * h, (c.z + stagger.z) * h};
            if (b >= 0) {
                int i = b * 512 + (c.x & 7) + 8 * (c.y & 7) + 64 * (c.z & 7);
                if (field >= 3) value = (field == 3 ? fields.smoke : fields.temperature)[i];
                else {
                    int f      = negative_face(i, field);
                    bool owned = grid.faces[f].left >= 0 || grid.faces[f].right >= 0;
                    uniform &= owned && grid.faces[f].group < 0;
                    value = owned ? fields.velocity[f] : side_velocity(grid, fields, i, field, -1);
                }
            } else if (field >= 3) {
                int i = locate(grid.layout, point);
                if (i >= 0) {
                    const float* values = field == 3 ? fields.smoke : fields.temperature;
                    if (grid.layout.blocks[i >> 9].level > level) {
                        float3 gradient = scalar_gradient(grid, values, i), c = center(grid.layout, i);
                        value = values[i] + (gradient.x * (point.x - c.x) + gradient.y * (point.y - c.y) + gradient.z * (point.z - c.z));
                    } else {
                        value = 0;
                        for (int child = 0; child < 8; ++child) {
                            float3 sub{point.x + (child & 1 ? .25f : -.25f) * h, point.y + (child & 2 ? .25f : -.25f) * h, point.z + (child & 4 ? .25f : -.25f) * h};
                            int j = locate(grid.layout, sub);
                            if (j >= 0) value += values[j] * .125f;
                        }
                    }
                }
            } else {
                float epsilon = grid.layout.h * .001f;
                int i         = locate(grid.layout, float3{point.x + (field == 0 ? epsilon : 0), point.y + (field == 1 ? epsilon : 0), point.z + (field == 2 ? epsilon : 0)});
                if (i >= 0 && grid.layout.blocks[i >> 9].level < level) {
                    value = 0;
                    for (int child = 0; child < 4; ++child) {
                        int a = (field + 1) % 3, b = (field + 2) % 3;
                        float u = (child & 1 ? .25f : -.25f) * h, v = (child & 2 ? .25f : -.25f) * h;
                        value += .25f * face_value(grid, fields, {point.x + (a == 0 ? u : 0) + (b == 0 ? v : 0), point.y + (a == 1 ? u : 0) + (b == 1 ? v : 0), point.z + (a == 2 ? u : 0) + (b == 2 ? v : 0)}, field, background);
                    }
                } else value = face_value(grid, fields, point, field, background);
            }
            int index                   = (block * 5 + field) * 1331 + local;
            output[index]               = value;
            grid.uniform_samples[index] = uniform;
        }
    }

    void advect(Grid grid, Fields destination, Fields carrier, float cell_size, float3 wind, float time, float dt, ::cuda::stream_ref stream) {
        grid.state.samples       = grid.samples[0];
        grid.transported.samples = grid.samples[2];
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3(grid.layout.count, 5)), ::cuda::block_dims<256>()), cache_samples, grid, grid.state, wind, grid.samples[0]);
        if (carrier.velocity == grid.state.velocity) carrier.samples = grid.samples[0];
        else {
            carrier.samples = grid.samples[1];
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3(grid.layout.count, 3)), ::cuda::block_dims<256>()), cache_samples, grid, carrier, wind, grid.samples[1]);
        }

        auto launch = [&]<BoundaryMode boundary_mode>() {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), transport_cells<false, boundary_mode>, grid, grid.transported, carrier, cell_size, wind, time, dt);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), transport_faces<false, boundary_mode>, grid, grid.transported, carrier, cell_size, wind, time, dt);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3(grid.layout.count, 5)), ::cuda::block_dims<256>()), cache_samples, grid, grid.transported, wind, grid.samples[2]);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), transport_cells<true, boundary_mode>, grid, destination, carrier, cell_size, wind, time, dt);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), transport_faces<true, boundary_mode>, grid, destination, carrier, cell_size, wind, time, dt);
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none: launch.template operator()<BoundaryMode::none>(); break;
        case BoundaryMode::stationary: launch.template operator()<BoundaryMode::stationary>(); break;
        case BoundaryMode::moving: launch.template operator()<BoundaryMode::moving>(); break;
        }
    }
}
