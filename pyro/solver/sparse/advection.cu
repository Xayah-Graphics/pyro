#include "advection.cuh"
#include <cuda/launch>

namespace pyro::sparse {
    struct ScalarReconstruction {
        float3 gradient;
        float predictor;
        float lower;
        float upper;
    };

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
                float dx = sample_boundary<boundary_mode>(grid, {p.x + h * .25f, p.y, p.z}, time) - sample_boundary<boundary_mode>(grid, {p.x - h * .25f, p.y, p.z}, time);
                float dy = sample_boundary<boundary_mode>(grid, {p.x, p.y + h * .25f, p.z}, time) - sample_boundary<boundary_mode>(grid, {p.x, p.y - h * .25f, p.z}, time);
                float dz = sample_boundary<boundary_mode>(grid, {p.x, p.y, p.z + h * .25f}, time) - sample_boundary<boundary_mode>(grid, {p.x, p.y, p.z - h * .25f}, time);
                gradient = {dx, dy, dz};
                float scale = (.01f * h - distance) / fmaxf(sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z), 1.e-20f);
                p = {p.x + gradient.x * scale, p.y + gradient.y * scale, p.z + gradient.z * scale};
                clipped = true;
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
        float background = axis == 0 ? wind.x : axis == 1 ? wind.y
                                                          : wind.z;
        float value;
        if constexpr (correct) {
            float lo = destination.velocity[f], hi = grid.velocity_scratch[f];
            value = grid.transported.velocity[f];
            if (lo == hi) value = lo;
            else if (boundary_mode == BoundaryMode::none || lo <= hi) {
                bool clipped;
                float3 reverse = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time, -dt, clipped);
                if (!clipped) {
                    float corrected = value + .5f * (grid.state.velocity[f] - sample(grid, grid.transported, reverse, axis, locate(grid.layout, reverse), background));
                    if (corrected >= lo && corrected <= hi) value = corrected;
                }
            }
        } else {
            bool clipped;
            float3 departure = trace<boundary_mode>(grid, carrier, cell_size, wind, p, time + dt, dt, clipped);
            float lo, hi;
            value = sample<true>(grid, grid.state, departure, axis, locate(grid.layout, departure), background, &lo, &hi);
            destination.velocity[f] = clipped ? 1 : lo;
            grid.velocity_scratch[f] = clipped ? -1 : hi;
        }
        output[f] = value;
    }

    __global__ void cache_velocity(Grid grid, Fields fields, float3 wind, float* output) {
        int block = blockIdx.x, axis = blockIdx.y;
        Block node = grid.layout.blocks[block];
        int level = node.level;
        float h = ldexpf(grid.layout.h, level);
        float3 stagger{axis == 0 ? 0.f : .5f, axis == 1 ? 0.f : .5f, axis == 2 ? 0.f : .5f};
        float background = axis == 0 ? wind.x : axis == 1 ? wind.y
                                                          : wind.z;
        for (int local = threadIdx.x; local < 1331; local += blockDim.x) {
            int3 c{node.coordinate.x * 8 + local % 11 - 1, node.coordinate.y * 8 + (local / 11) % 11 - 1, node.coordinate.z * 8 + local / 121 - 1};
            int b = (c.x >> 3) == node.coordinate.x && (c.y >> 3) == node.coordinate.y && (c.z >> 3) == node.coordinate.z ? block : find_block(grid.layout, {c.x >> 3, c.y >> 3, c.z >> 3}, level);
            float value = background;
            bool uniform = b >= 0;
            float3 point{(c.x + stagger.x) * h, (c.y + stagger.y) * h, (c.z + stagger.z) * h};
            if (b >= 0) {
                int i = b * 512 + (c.x & 7) + 8 * (c.y & 7) + 64 * (c.z & 7);
                int f = negative_face(i, axis);
                bool owned = grid.faces[f].left >= 0 || grid.faces[f].right >= 0;
                uniform &= owned && grid.faces[f].group < 0;
                value = owned ? fields.velocity[f] : side_velocity(grid, fields, i, axis, -1);
            } else {
                float epsilon = grid.layout.h * .001f;
                int i = locate(grid.layout, float3{point.x + (axis == 0 ? epsilon : 0), point.y + (axis == 1 ? epsilon : 0), point.z + (axis == 2 ? epsilon : 0)});
                if (i >= 0 && grid.layout.blocks[i >> 9].level < level) {
                    value = 0;
                    for (int child = 0; child < 4; ++child) {
                        int a = (axis + 1) % 3, b = (axis + 2) % 3;
                        float u = (child & 1 ? .25f : -.25f) * h, v = (child & 2 ? .25f : -.25f) * h;
                        value += .25f * face_value(grid, fields, {point.x + (a == 0 ? u : 0) + (b == 0 ? v : 0), point.y + (a == 1 ? u : 0) + (b == 1 ? v : 0), point.z + (a == 2 ? u : 0) + (b == 2 ? v : 0)}, axis, background);
                    }
                } else value = face_value(grid, fields, point, axis, background);
            }
            int index = (block * 3 + axis) * 1331 + local;
            output[index] = value;
            grid.uniform_velocity_samples[index] = uniform;
        }
    }

    template <BoundaryMode boundary_mode>
    __global__ void reconstruct_scalars(Grid grid, Fields original, Fields carrier, float dt, ScalarReconstruction* smoke, ScalarReconstruction* temperature) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512 || !grid.fluid[i]) return;
        float h = spacing(grid.layout, i);
        float q[2]{original.smoke[i], original.temperature[i]};
        float lower[2]{q[0], q[1]}, upper[2]{q[0], q[1]}, slopes[2][3];
        bool varying = false;
        for (int a = 0; a < 3; ++a) {
            float adjacent[2][2], distance[2]{h, h};
            int local = (i >> (3 * a)) & 7, stride = 1 << (3 * a);
            for (int side = 0; side < 2; ++side) {
                int sign = side ? 1 : -1;
                if (side ? local < 7 : local > 0) {
                    int j = i + sign * stride;
                    bool open = grid.fluid[j] != 0;
                    if constexpr (boundary_mode != BoundaryMode::none) open &= grid.faces[negative_face(i + (side ? stride : 0), a)].area > 0;
                    adjacent[side][0] = open ? original.smoke[j] : q[0];
                    adjacent[side][1] = open ? original.temperature[j] : q[1];
                } else {
                    auto links = cell_faces(grid, i, a, sign);
                    float total = 0, density = 0, heat = 0, separation = 0;
                    for (int f : links.face)
                        if (f >= 0) {
                            Face face = grid.faces[f];
                            int j = face.left == i ? face.right : face.left;
                            if (j >= 0 && grid.fluid[j]) {
                                total += face.area;
                                density += face.area * original.smoke[j];
                                heat += face.area * original.temperature[j];
                                separation += face.area * face.distance;
                            }
                        }
                    adjacent[side][0] = total > 0 ? density / total : q[0];
                    adjacent[side][1] = total > 0 ? heat / total : q[1];
                    if (total > 0) distance[side] = separation / total;
                }
            }
            for (int field = 0; field < 2; ++field) {
                float l = adjacent[0][field], r = adjacent[1][field];
                lower[field] = fminf(lower[field], fminf(l, r));
                upper[field] = fmaxf(upper[field], fmaxf(l, r));
                float dl = (q[field] - l) / distance[0], dr = (r - q[field]) / distance[1];
                float central = (dl * distance[1] + dr * distance[0]) / (distance[0] + distance[1]);
                slopes[field][a] = dl * dr > 0 ? copysignf(fminf(fabsf(central), 2 * fminf(fabsf(dl), fabsf(dr))), dl) : 0;
                varying |= slopes[field][a] != 0;
            }
        }
        float3 velocity{};
        if (varying) velocity = {.5f * (side_velocity<boundary_mode>(grid, carrier, i, 0, -1) + side_velocity<boundary_mode>(grid, carrier, i, 0, 1)), .5f * (side_velocity<boundary_mode>(grid, carrier, i, 1, -1) + side_velocity<boundary_mode>(grid, carrier, i, 1, 1)), .5f * (side_velocity<boundary_mode>(grid, carrier, i, 2, -1) + side_velocity<boundary_mode>(grid, carrier, i, 2, 1))};
        for (int field = 0; field < 2; ++field) {
            float3 gradient{slopes[field][0], slopes[field][1], slopes[field][2]};
            (field ? temperature : smoke)[i] = {gradient, q[field] - .5f * dt * (velocity.x * gradient.x + velocity.y * gradient.y + velocity.z * gradient.z), lower[field], upper[field]};
        }
    }

    template <BoundaryMode boundary_mode>
    __global__ void scalar_fluxes(Grid grid, Fields carrier, const ScalarReconstruction* smoke, const ScalarReconstruction* temperature) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.left < 0 && face.right < 0) return;
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        float area = boundary_mode == BoundaryMode::none ? h * h : face.area;
        float speed = carrier.velocity[f];
        int upwind = speed >= 0 ? face.left : face.right;
        float flux[2]{};
        if (area > 0 && speed != 0 && upwind >= 0 && grid.fluid[upwind]) {
            float3 cell_center = center(grid.layout, upwind);
            for (int field = 0; field < 2; ++field) {
                auto value = (field ? temperature : smoke)[upwind];
                float reconstructed = value.predictor + value.gradient.x * (p.x - cell_center.x) + value.gradient.y * (p.y - cell_center.y) + value.gradient.z * (p.z - cell_center.z);
                flux[field] = speed * area * fminf(value.upper, fmaxf(value.lower, reconstructed));
            }
        }
        grid.velocity_scratch[f] = flux[0];
        grid.transported.velocity[f] = flux[1];
    }

    __global__ void conservative_cells(Grid grid, Fields original, Fields destination, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        if (!grid.fluid[i]) {
            destination.smoke[i] = destination.temperature[i] = 0;
            return;
        }
        float flux[2]{};
        float h = spacing(grid.layout, i);
        for (int a = 0; a < 3; ++a)
            for (int sign : {-1, 1}) {
                auto links = cell_faces(grid, i, a, sign);
                for (int face : links.face)
                    if (face >= 0) {
                        flux[0] += sign * grid.velocity_scratch[face];
                        flux[1] += sign * grid.transported.velocity[face];
                    }
            }

        destination.smoke[i] = original.smoke[i] - dt / (h * h * h) * flux[0];
        destination.temperature[i] = original.temperature[i] - dt / (h * h * h) * flux[1];
    }

    void advect(Grid grid, Fields destination, Fields carrier, float cell_size, float3 wind, float time, float dt, ::cuda::stream_ref stream) {
        grid.state.velocity_samples = grid.velocity_samples[0];
        grid.transported.velocity_samples = grid.velocity_samples[2];
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3(grid.layout.count, 3)), ::cuda::block_dims<256>()), cache_velocity, grid, grid.state, wind, grid.velocity_samples[0]);
        if (carrier.velocity == grid.state.velocity) carrier.velocity_samples = grid.velocity_samples[0];
        else {
            carrier.velocity_samples = grid.velocity_samples[1];
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3(grid.layout.count, 3)), ::cuda::block_dims<256>()), cache_velocity, grid, carrier, wind, grid.velocity_samples[1]);
        }

        auto* smoke = reinterpret_cast<ScalarReconstruction*>(grid.velocity_samples[0]);
        auto* temperature = reinterpret_cast<ScalarReconstruction*>(grid.velocity_samples[1]);
        auto launch = [&]<BoundaryMode boundary_mode>() {
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), transport_faces<false, boundary_mode>, grid, destination, carrier, cell_size, wind, time, dt);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3(grid.layout.count, 3)), ::cuda::block_dims<256>()), cache_velocity, grid, grid.transported, wind, grid.velocity_samples[2]);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), transport_faces<true, boundary_mode>, grid, destination, carrier, cell_size, wind, time, dt);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 512), reconstruct_scalars<boundary_mode>, grid, grid.state, carrier, dt, smoke, temperature);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), scalar_fluxes<boundary_mode>, grid, carrier, smoke, temperature);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 512), conservative_cells, grid, grid.state, destination, dt);
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none:
            launch.template operator()<BoundaryMode::none>();
            break;
        case BoundaryMode::stationary:
            launch.template operator()<BoundaryMode::stationary>();
            break;
        case BoundaryMode::moving:
            launch.template operator()<BoundaryMode::moving>();
            break;
        }
    }
}
