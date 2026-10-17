#include "advection.cuh"

#include <cuda/launch>

namespace pyro::dense {
    struct ScalarReconstruction {
        float3 gradient;
        float predictor;
        float lower;
        float upper;
    };

    template <BoundaryMode boundary_mode>
    __device__ float3 trace(Grid grid, Fields carrier, float3 p, float3 wind, float h, float end_time, float dt, bool& clipped) {
        Layout layout = grid.layout;
        clipped = false;
        float first_distance = 0;
        if constexpr (boundary_mode != BoundaryMode::none) {
            float3 world{grid.origin.x + p.x * h, grid.origin.y + p.y * h, grid.origin.z + p.z * h};
            float distance = sample_boundary<boundary_mode>(grid, world, end_time, h);
            first_distance = distance;
            if (distance < 0.01f * h) {
                float e = 0.25f * h;
                float3 gradient{sample_boundary<boundary_mode>(grid, {world.x + e, world.y, world.z}, end_time, h) - sample_boundary<boundary_mode>(grid, {world.x - e, world.y, world.z}, end_time, h), sample_boundary<boundary_mode>(grid, {world.x, world.y + e, world.z}, end_time, h) - sample_boundary<boundary_mode>(grid, {world.x, world.y - e, world.z}, end_time, h), sample_boundary<boundary_mode>(grid, {world.x, world.y, world.z + e}, end_time, h) - sample_boundary<boundary_mode>(grid, {world.x, world.y, world.z - e}, end_time, h)};
                float scale = (0.01f * h - distance) / (h * fmaxf(sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z), 1.0e-20f));
                p = {p.x + gradient.x * scale, p.y + gradient.y * scale, p.z + gradient.z * scale};
                clipped = true;
            }
        }
        float3 first = sample_velocity(layout, carrier, p, wind);
        float3 middle = sample_velocity(layout, carrier, {p.x - first.x * (0.5f * dt / h), p.y - first.y * (0.5f * dt / h), p.z - first.z * (0.5f * dt / h)}, wind);
        float3 departure{p.x - middle.x * (dt / h), p.y - middle.y * (dt / h), p.z - middle.z * (dt / h)};
        if constexpr (boundary_mode == BoundaryMode::none) return departure;
        float3 delta{departure.x - p.x, departure.y - p.y, departure.z - p.z};
        float bound = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) * h + (boundary_mode == BoundaryMode::moving ? grid.collider_speed * fabsf(dt) : 0);
        if (bound < 1.0e-12f) return departure;
        float fraction = 0.0f;
        int first_step = 0;
        if (!clipped) {
            fraction = .5f * first_distance / bound;
            if (fraction >= 1) return departure;
            first_step = 1;
        }
        for (int step = first_step; step < 64; ++step) {
            float3 position{p.x + delta.x * fraction, p.y + delta.y * fraction, p.z + delta.z * fraction};
            float distance = sample_boundary<boundary_mode>(grid, {grid.origin.x + position.x * h, grid.origin.y + position.y * h, grid.origin.z + position.z * h}, end_time - dt * fraction, h);
            if (distance < 0.001f * h) {
                clipped = true;
                return position;
            }
            fraction += 0.5f * distance / bound;
            if (fraction >= 1.0f) return departure;
        }
        clipped = true;
        return {p.x + delta.x * fraction, p.y + delta.y * fraction, p.z + delta.z * fraction};
    }

    template <bool correct, BoundaryMode boundary_mode>
    __global__ void transport_velocity(Grid grid, float cell_size, float3 wind, Fields destination, Fields original, Fields carrier, float time, float dt) {
        int id = static_cast<int>(blockIdx.x), axis = blockIdx.y;
        for (int local = threadIdx.x; local < (boundary_mode == BoundaryMode::none ? tile_cells : threads); local += blockDim.x) {
            int i = id * (boundary_mode == BoundaryMode::none ? tile_cells : threads) + local;
            float* output = correct ? destination.velocity[axis] : grid.transported.velocity[axis];
            if (!face_valid(grid.layout, cell_coordinate(grid.layout, i), axis)) {
                output[i] = (axis == 0 ? wind.x : axis == 1 ? wind.y
                                                            : wind.z);
                continue;
            }
            if constexpr (boundary_mode != BoundaryMode::none)
                if (grid.aperture[axis][i] == 0.0f) {
                    output[i] = grid.solid_velocity[axis][i];
                    continue;
                }
            int3 c = cell_coordinate(grid.layout, i);
            float3 stagger{0.5f, 0.5f, 0.5f};
            if (axis == 0) stagger.x = 0.0f;
            if (axis == 1) stagger.y = 0.0f;
            if (axis == 2) stagger.z = 0.0f;
            float3 p{c.x + stagger.x, c.y + stagger.y, c.z + stagger.z};
            const float* input = original.velocity[axis];
            float background = (axis == 0 ? wind.x : axis == 1 ? wind.y
                                                               : wind.z);
            float transported;
            if constexpr (correct) {
                float lo = destination.velocity[axis][i], hi = grid.velocity_scratch[axis][i];
                transported = grid.transported.velocity[axis][i];
                if (lo == hi) transported = lo;
                else if (boundary_mode == BoundaryMode::none || lo <= hi) {
                    bool clipped;
                    float3 reverse = trace<boundary_mode>(grid, carrier, p, wind, cell_size, time, -dt, clipped);
                    if (!clipped) {
                        float corrected = transported + 0.5f * (input[i] - sample(grid.layout, grid.transported.velocity[axis], {reverse.x - stagger.x, reverse.y - stagger.y, reverse.z - stagger.z}, background));
                        if (corrected >= lo && corrected <= hi) transported = corrected;
                    }
                }
            } else {
                bool clipped;
                float3 departure = trace<boundary_mode>(grid, carrier, p, wind, cell_size, time + dt, dt, clipped);
                float lo, hi;
                transported = sample<true>(grid.layout, input, {departure.x - stagger.x, departure.y - stagger.y, departure.z - stagger.z}, background, &lo, &hi);
                destination.velocity[axis][i] = clipped ? 1 : lo;
                grid.velocity_scratch[axis][i] = clipped ? -1 : hi;
            }
            output[i] = transported;
        }
    }

    template <BoundaryMode boundary_mode>
    __global__ void reconstruct_scalars(Grid grid, Fields original, Fields carrier, float cell_size, float dt, ScalarReconstruction* smoke, ScalarReconstruction* temperature) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.cells || !grid.fluid[i]) return;
        float h = cell_size;
        ScalarReconstruction result[2];
        bool varying = false;
        for (int field = 0; field < 2; ++field) {
            const float* values = field ? original.temperature : original.smoke;
            float q = values[i], lower = q, upper = q, slopes[3];
            for (int a = 0; a < 3; ++a) {
                int left = neighbor(grid.layout, i, a, -1), right = neighbor(grid.layout, i, a, 1);
                bool left_open = left >= 0 && grid.fluid[left], right_open = right >= 0 && grid.fluid[right];
                if constexpr (boundary_mode != BoundaryMode::none) {
                    left_open &= grid.aperture[a][i] > 0;
                    right_open &= grid.aperture[a][right] > 0;
                }
                float l = left_open ? values[left] : q;
                float r = right_open ? values[right] : q;
                lower = fminf(lower, fminf(l, r));
                upper = fmaxf(upper, fmaxf(l, r));
                float dl = (q - l) / h, dr = (r - q) / h;
                float central = .5f * (dl + dr);
                slopes[a] = dl * dr > 0 ? copysignf(fminf(fabsf(central), 2 * fminf(fabsf(dl), fabsf(dr))), dl) : 0;
            }
            result[field] = {{slopes[0], slopes[1], slopes[2]}, q, lower, upper};
            varying |= slopes[0] != 0 || slopes[1] != 0 || slopes[2] != 0;
        }
        if (varying) {
            float3 velocity{.5f * (carrier.velocity[0][i] + carrier.velocity[0][neighbor(grid.layout, i, 0, 1)]), .5f * (carrier.velocity[1][i] + carrier.velocity[1][neighbor(grid.layout, i, 1, 1)]), .5f * (carrier.velocity[2][i] + carrier.velocity[2][neighbor(grid.layout, i, 2, 1)])};
            for (auto& value : result) value.predictor -= .5f * dt * (velocity.x * value.gradient.x + velocity.y * value.gradient.y + velocity.z * value.gradient.z);
        }
        smoke[i] = result[0];
        temperature[i] = result[1];
    }

    template <BoundaryMode boundary_mode>
    __global__ void scalar_fluxes(Grid grid, Fields carrier, float cell_size, const ScalarReconstruction* smoke, const ScalarReconstruction* temperature) {
        int f = blockIdx.x * blockDim.x + threadIdx.x, axis = blockIdx.y;
        if (f >= grid.layout.cells) return;
        auto c = cell_coordinate(grid.layout, f);
        if (!face_valid(grid.layout, c, axis)) return;
        float h = cell_size;
        float area = (boundary_mode == BoundaryMode::none ? 1.f : grid.aperture[axis][f]) * h * h;
        float speed = carrier.velocity[axis][f];
        int upwind = speed >= 0 ? neighbor(grid.layout, f, axis, -1) : f;
        float flux[2]{};
        if (area > 0 && speed != 0 && upwind >= 0 && grid.fluid[upwind]) {
            float offset = speed >= 0 ? .5f * h : -.5f * h;
            for (int field = 0; field < 2; ++field) {
                auto value = (field ? temperature : smoke)[upwind];
                float slope = axis == 0 ? value.gradient.x : axis == 1 ? value.gradient.y : value.gradient.z;
                float reconstructed = value.predictor + slope * offset;
                flux[field] = speed * area * fminf(value.upper, fmaxf(value.lower, reconstructed));
            }
        }
        grid.velocity_scratch[axis][f] = flux[0];
        grid.transported.velocity[axis][f] = flux[1];
    }

    __global__ void conservative_cells(Grid grid, Fields original, Fields destination, float cell_size, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.cells) return;
        if (!grid.fluid[i]) {
            destination.smoke[i] = destination.temperature[i] = 0;
            return;
        }
        float flux[2]{};
        float h = cell_size;

        for (int a = 0; a < 3; ++a) {
            int high = neighbor(grid.layout, i, a, 1);
            flux[0] += grid.velocity_scratch[a][high] - grid.velocity_scratch[a][i];
            flux[1] += grid.transported.velocity[a][high] - grid.transported.velocity[a][i];
        }

        destination.smoke[i] = original.smoke[i] - dt / (h * h * h) * flux[0];
        destination.temperature[i] = original.temperature[i] - dt / (h * h * h) * flux[1];
    }

    void advect(Grid grid, float cell_size, float3 wind, Fields destination, Fields original, Fields carrier, float start_time, float dt, ::cuda::stream_ref stream) {
        auto* smoke = reinterpret_cast<ScalarReconstruction*>(grid.scalar_scratch[0]);
        auto* temperature = reinterpret_cast<ScalarReconstruction*>(grid.scalar_scratch[1]);
        auto launch = [&]<BoundaryMode boundary_mode>() {
            constexpr int count = boundary_mode == BoundaryMode::none ? tile_cells : threads;
            dim3 velocity_blocks(grid.layout.cells / count, 3);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(velocity_blocks), ::cuda::block_dims<threads>()), transport_velocity<false, boundary_mode>, grid, cell_size, wind, destination, original, carrier, start_time, dt);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(velocity_blocks), ::cuda::block_dims<threads>()), transport_velocity<true, boundary_mode>, grid, cell_size, wind, destination, original, carrier, start_time, dt);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.cells), reconstruct_scalars<boundary_mode>, grid, original, carrier, cell_size, dt, smoke, temperature);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3((grid.layout.cells + 255) / 256, 3)), ::cuda::block_dims<256>()), scalar_fluxes<boundary_mode>, grid, carrier, cell_size, smoke, temperature);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.cells), conservative_cells, grid, original, destination, cell_size, dt);
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
