#include "advection.cuh"
#include <cuda/launch>

namespace pyro::sparse {
    template <BoundaryMode boundary_mode>
    __device__ float3 trace(Grid grid, Fields carrier, float cell_size, float3 wind, float3 p, float time, float dt, bool& clipped) {
        clipped = false;
        float first_distance = 0;
        float h = cell_size;
        if constexpr (boundary_mode != BoundaryMode::none) {
            float distance = sample_boundary<boundary_mode>(grid, p, time);
            first_distance = distance;
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
        int first_step = 0;
        if (!clipped) {
            fraction = .5f * first_distance / bound;
            if (fraction >= 1) return departure;
            first_step = 1;
        }
        for (int step = first_step; step < 64; ++step) {
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

    template <bool bounds, bool uniform_output = false>
    __device__ float2 native_scalars(Grid grid, Fields fields, float3 p, int anchor, float2* lower = nullptr, float2* upper = nullptr, bool* uniform = nullptr) {
        int block = anchor >> 9;
        Block node = grid.layout.blocks[block];
        float inverse_h = 1.f / spacing(grid.layout, anchor);
        float3 q{p.x * inverse_h - .5f, p.y * inverse_h - .5f, p.z * inverse_h - .5f};
        int3 base{int(floorf(q.x)), int(floorf(q.y)), int(floorf(q.z))};
        float3 t{q.x - base.x, q.y - base.y, q.z - base.z};
        int first = (block * 5 + 3) * 1331 + base.x - node.coordinate.x * 8 + 1 + 11 * (base.y - node.coordinate.y * 8 + 1) + 121 * (base.z - node.coordinate.z * 8 + 1);
        float2 sum{}, lo{1.e30f, 1.e30f}, hi{-1.e30f, -1.e30f};
        bool same_level = true;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    int index = first + x + 11 * y + 121 * z;
                    float a = fields.samples[index], b = fields.samples[index + 1331];
                    sum.x += a * (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y) * (z ? t.z : 1 - t.z);
                    sum.y += b * (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y) * (z ? t.z : 1 - t.z);
                    if constexpr (uniform_output) same_level &= grid.uniform_samples[index] != 0;
                    if constexpr (bounds) {
                        lo.x = fminf(lo.x, a); lo.y = fminf(lo.y, b);
                        hi.x = fmaxf(hi.x, a); hi.y = fmaxf(hi.y, b);
                    }
                }
        if constexpr (bounds) { *lower = lo; *upper = hi; }
        if constexpr (uniform_output) *uniform = same_level;
        return sum;
    }

    template <bool bounds = false>
    __device__ float2 sample_scalars(Grid grid, Fields fields, float3 p, int anchor, float2* lower = nullptr, float2* upper = nullptr) {
        if (anchor >= 0) {
            bool uniform;
            float2 value = native_scalars<bounds, true>(grid, fields, p, anchor, lower, upper, &uniform);
            if (uniform) return value;
        }
        float h = .5f * grid.layout.h, inverse_h = 1.f / h;
        float3 q{p.x * inverse_h, p.y * inverse_h, p.z * inverse_h};
        int3 base{int(floorf(q.x)), int(floorf(q.y)), int(floorf(q.z))};
        float3 t{q.x - base.x, q.y - base.y, q.z - base.z};
        float2 sum{}, lo{1.e30f, 1.e30f}, hi{-1.e30f, -1.e30f};
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    float3 point{(base.x + x) * h, (base.y + y) * h, (base.z + z) * h};
                    int i = locate(grid.layout, point);
                    if (i < 0)
                        for (int corner = 1; corner < 8 && i < 0; ++corner) i = locate(grid.layout, float3{point.x - (corner & 1 ? .002f : 0) * h, point.y - (corner & 2 ? .002f : 0) * h, point.z - (corner & 4 ? .002f : 0) * h});
                    float2 value = i < 0 ? float2{} : native_scalars<false>(grid, fields, point, i);
                    sum.x += value.x * (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y) * (z ? t.z : 1 - t.z);
                    sum.y += value.y * (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y) * (z ? t.z : 1 - t.z);
                    if constexpr (bounds) {
                        lo.x = fminf(lo.x, value.x); lo.y = fminf(lo.y, value.y);
                        hi.x = fmaxf(hi.x, value.x); hi.y = fmaxf(hi.y, value.y);
                    }
                }
        if constexpr (bounds) { *lower = lo; *upper = hi; }
        return sum;
    }

    template <bool correct, BoundaryMode boundary_mode>
    __global__ void transport_cells(Grid grid, Fields destination, Fields carrier, float cell_size, float3 wind, float time, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        auto output = correct ? destination : grid.transported;
        if constexpr (boundary_mode != BoundaryMode::none)
            if (!grid.fluid[i]) {
                output.smoke[i] = output.temperature[i] = 0;
                return;
            }
        float3 p = center(grid.layout, i);
        float2 value;
        if constexpr (correct) {
            bool clipped = boundary_mode != BoundaryMode::none && destination.smoke[i] > grid.upper.smoke[i];
            value = {grid.transported.smoke[i], grid.transported.temperature[i]};
            bool constant = __float_as_uint(destination.smoke[i]) == __float_as_uint(grid.upper.smoke[i]) && __float_as_uint(destination.temperature[i]) == __float_as_uint(grid.upper.temperature[i]) && __float_as_uint(value.x) == __float_as_uint(destination.smoke[i]) && __float_as_uint(value.y) == __float_as_uint(destination.temperature[i]);
            if (!clipped && !constant) {
                float3 reverse = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time, -dt, clipped);
                if (!clipped) {
                    float2 backward = sample_scalars(grid, grid.transported, reverse, locate(grid.layout, reverse));
                    value.x = fminf(grid.upper.smoke[i], fmaxf(destination.smoke[i], value.x + .5f * (grid.state.smoke[i] - backward.x)));
                    value.y = fminf(grid.upper.temperature[i], fmaxf(destination.temperature[i], value.y + .5f * (grid.state.temperature[i] - backward.y)));
                }
            }
        } else {
            bool clipped;
            float3 departure = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time + dt, dt, clipped);
            float2 lo, hi;
            value = sample_scalars<true>(grid, grid.state, departure, locate(grid.layout, departure), &lo, &hi);
            destination.smoke[i] = clipped ? 1 : lo.x;
            destination.temperature[i] = lo.y;
            grid.upper.smoke[i] = clipped ? -1 : hi.x;
            grid.upper.temperature[i] = hi.y;
        }
        output.smoke[i] = value.x;
        output.temperature[i] = value.y;
    }

    template <bool correct, BoundaryMode boundary_mode>
    __global__ void transport_faces(Grid grid, Fields destination, Fields carrier, float cell_size, float3 wind, float time, float dt) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        float* output = correct ? destination.velocity : grid.transported.velocity;
        if (face.left < 0 && face.right < 0) {
            output[f] = 0;
            return;
        }
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        if constexpr (boundary_mode != BoundaryMode::none)
            if (face.area <= 0) {
                output[f] = face.solid_flux / (h * h);
                return;
            }
        float background = axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z;
        float value;
        if constexpr (correct) {
            float lo = destination.velocity[f], hi = grid.upper.velocity[f];
            value = grid.transported.velocity[f];
            bool constant = __float_as_uint(lo) == __float_as_uint(hi) && __float_as_uint(value) == __float_as_uint(lo);
            if (!constant && (boundary_mode == BoundaryMode::none || lo <= hi)) {
                bool clipped;
                float3 reverse = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time, -dt, clipped);
                if (!clipped) value = fminf(hi, fmaxf(lo, value + .5f * (grid.state.velocity[f] - sample(grid, grid.transported, reverse, axis, locate(grid.layout, reverse), background))));
            }
        } else {
            bool clipped;
            float3 departure = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time + dt, dt, clipped);
            float lo, hi;
            value = sample<true>(grid, grid.state, departure, axis, locate(grid.layout, departure), background, &lo, &hi);
            destination.velocity[f] = clipped ? 1 : lo;
            grid.upper.velocity[f] = clipped ? -1 : hi;
        }
        output[f] = value;
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
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), transport_cells<false, boundary_mode>, grid, destination, carrier, cell_size, wind, time, dt);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), transport_faces<false, boundary_mode>, grid, destination, carrier, cell_size, wind, time, dt);
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
