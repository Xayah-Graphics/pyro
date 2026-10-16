#include "advection.cuh"

#include <cuda/launch>

namespace pyro::dense {
    template <BoundaryMode boundary_mode>
    __device__ float3 trace(Grid grid, Fields carrier, float3 p, float3 wind, float h, float end_time, float dt, bool& clipped) {
        Layout layout = grid.layout;
        clipped       = false;
        if constexpr (boundary_mode != BoundaryMode::none) {
            float3 world{grid.origin.x + p.x * h, grid.origin.y + p.y * h, grid.origin.z + p.z * h};
            float distance = sample_boundary<boundary_mode>(grid, world, end_time, h);
            if (distance < 0.01f * h) {
                float e = 0.25f * h;
                float3 gradient{sample_boundary<boundary_mode>(grid, {world.x + e, world.y, world.z}, end_time, h) - sample_boundary<boundary_mode>(grid, {world.x - e, world.y, world.z}, end_time, h), sample_boundary<boundary_mode>(grid, {world.x, world.y + e, world.z}, end_time, h) - sample_boundary<boundary_mode>(grid, {world.x, world.y - e, world.z}, end_time, h), sample_boundary<boundary_mode>(grid, {world.x, world.y, world.z + e}, end_time, h) - sample_boundary<boundary_mode>(grid, {world.x, world.y, world.z - e}, end_time, h)};
                float scale = (0.01f * h - distance) / (h * fmaxf(sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z), 1.0e-20f));
                p           = {p.x + gradient.x * scale, p.y + gradient.y * scale, p.z + gradient.z * scale};
                clipped     = true;
            }
        }
        float3 first  = sample_velocity(layout, carrier, p, wind);
        float3 middle = sample_velocity(layout, carrier, {p.x - first.x * (0.5f * dt / h), p.y - first.y * (0.5f * dt / h), p.z - first.z * (0.5f * dt / h)}, wind);
        float3 departure{p.x - middle.x * (dt / h), p.y - middle.y * (dt / h), p.z - middle.z * (dt / h)};
        if constexpr (boundary_mode == BoundaryMode::none) return departure;
        float3 delta{departure.x - p.x, departure.y - p.y, departure.z - p.z};
        float bound = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) * h + (boundary_mode == BoundaryMode::moving ? grid.collider_speed * fabsf(dt) : 0);
        if (bound < 1.0e-12f) return departure;
        float fraction = 0.0f;
        for (int step = 0; step < 64; ++step) {
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

    template <bool bounds = false>
    __device__ float2 sample_scalars(Layout layout, Fields fields, float3 position, float2* lower = nullptr, float2* upper = nullptr) {
        int3 p{static_cast<int>(floorf(position.x)), static_cast<int>(floorf(position.y)), static_cast<int>(floorf(position.z))};
        float3 t{position.x - p.x, position.y - p.y, position.z - p.z};
        float2 result{}, lo{1.0e30f, 1.0e30f}, hi{-1.0e30f, -1.0e30f};
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    int i       = cell_index(layout, {p.x + x, p.y + y, p.z + z});
                    float smoke = value(fields.smoke, i), heat = value(fields.temperature, i);
                    result.x += smoke * (x ? t.x : 1.0f - t.x) * (y ? t.y : 1.0f - t.y) * (z ? t.z : 1.0f - t.z);
                    result.y += heat * (x ? t.x : 1.0f - t.x) * (y ? t.y : 1.0f - t.y) * (z ? t.z : 1.0f - t.z);
                    if constexpr (bounds) {
                        lo.x = fminf(lo.x, smoke);
                        lo.y = fminf(lo.y, heat);
                        hi.x = fmaxf(hi.x, smoke);
                        hi.y = fmaxf(hi.y, heat);
                    }
                }
        if constexpr (bounds) *lower = lo;
        if constexpr (bounds) *upper = hi;
        return result;
    }

    template <bool correct, BoundaryMode boundary_mode>
    __global__ void transport_velocity(Grid grid, float cell_size, float3 wind, Fields destination, Fields original, Fields carrier, float time, float dt) {
        int id = static_cast<int>(blockIdx.x), axis = blockIdx.y;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i         = id * tile_cells + local;
            float* output = destination.velocity[axis];
            if (!face_valid(grid.layout, cell_coordinate(grid.layout, i), axis)) {
                output[i] = (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z);
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
            float background   = (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z);
            bool clipped;
            float3 departure = trace<boundary_mode>(grid, carrier, p, wind, cell_size, time + dt, dt, clipped);
            float lo, hi;
            float transported = sample<correct>(grid.layout, input, {departure.x - stagger.x, departure.y - stagger.y, departure.z - stagger.z}, background, &lo, &hi);
            if (correct && !clipped) {
                float3 reverse = trace<boundary_mode>(grid, carrier, p, wind, cell_size, time, -dt, clipped);
                if (!clipped) transported = fminf(hi, fmaxf(lo, transported + 0.5f * (input[i] - sample(grid.layout, grid.transported.velocity[axis], {reverse.x - stagger.x, reverse.y - stagger.y, reverse.z - stagger.z}, background))));
            }
            output[i] = transported;
        }
    }

    template <bool correct, BoundaryMode boundary_mode>
    __global__ void transport_scalars(Grid grid, float cell_size, float3 wind, Fields destination, Fields original, Fields carrier, float time, float dt) {
        int id = static_cast<int>(blockIdx.x);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = id * tile_cells + local;
            if (!grid.fluid[i]) {
                destination.smoke[i]       = 0.0f;
                destination.temperature[i] = 0.0f;
                continue;
            }
            int3 c = cell_coordinate(grid.layout, i);
            float3 stagger{0.5f, 0.5f, 0.5f}, p{c.x + 0.5f, c.y + 0.5f, c.z + 0.5f};
            bool clipped;
            float3 departure = trace<boundary_mode>(grid, carrier, p, wind, cell_size, time + dt, dt, clipped);
            float2 lo, hi;
            float2 transported = sample_scalars<correct>(grid.layout, original, {departure.x - stagger.x, departure.y - stagger.y, departure.z - stagger.z}, &lo, &hi);
            if (correct && !clipped) {
                float3 reverse = trace<boundary_mode>(grid, carrier, p, wind, cell_size, time, -dt, clipped);
                if (!clipped) {
                    float2 backward = sample_scalars(grid.layout, grid.transported, {reverse.x - stagger.x, reverse.y - stagger.y, reverse.z - stagger.z});
                    transported.x   = fminf(hi.x, fmaxf(lo.x, transported.x + 0.5f * (original.smoke[i] - backward.x)));
                    transported.y   = fminf(hi.y, fmaxf(lo.y, transported.y + 0.5f * (original.temperature[i] - backward.y)));
                }
            }
            destination.smoke[i]       = transported.x;
            destination.temperature[i] = transported.y;
        }
    }

    void advect(Grid grid, float cell_size, float3 wind, Fields destination, Fields original, Fields carrier, float start_time, float dt, ::cuda::stream_ref stream) {
        dim3 velocity_blocks(grid.layout.blocks, 3);
        auto launch = [&]<BoundaryMode boundary_mode>() {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(velocity_blocks), ::cuda::block_dims<threads>()), transport_velocity<false, boundary_mode>, grid, cell_size, wind, grid.transported, original, carrier, start_time, dt);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), transport_scalars<false, boundary_mode>, grid, cell_size, wind, grid.transported, original, carrier, start_time, dt);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(velocity_blocks), ::cuda::block_dims<threads>()), transport_velocity<true, boundary_mode>, grid, cell_size, wind, destination, original, carrier, start_time, dt);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), transport_scalars<true, boundary_mode>, grid, cell_size, wind, destination, original, carrier, start_time, dt);
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none: launch.template operator()<BoundaryMode::none>(); break;
        case BoundaryMode::stationary: launch.template operator()<BoundaryMode::stationary>(); break;
        case BoundaryMode::moving: launch.template operator()<BoundaryMode::moving>(); break;
        }
    }

}
