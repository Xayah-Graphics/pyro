#include "statistics.cuh"
#include <cuda/launch>
#include <solver/common/scene.cuh>

namespace pyro::sparse {
    template <bool full, BoundaryMode boundary_mode>
    __global__ void measure_blocks(Grid grid, ForcesConfig config, double* results) {
        constexpr int count = full ? 6 : 2;
        __shared__ double sums[count][256];
        int b        = blockIdx.x;
        double fluid = 0, divergence = 0, speed = 0, smoke = 0, heat = 0, acceleration = 0;
        for (int local = threadIdx.x; local < 512; local += blockDim.x) {
            int i = b * 512 + local;
            if (!grid.fluid[i]) continue;
            float div = boundary_mode == BoundaryMode::moving ? -grid.solid_divergence[i] : 0, velocity_bound = 0;
            if constexpr (full) {
                double h = spacing(grid.layout, i), volume = h * h * h;
                ++fluid;
                smoke += grid.state.smoke[i] * volume;
                heat += grid.state.temperature[i] * volume;
            }
            for (int a = 0; a < 3; ++a) {
                if constexpr (full) div += (side_velocity<boundary_mode>(grid, grid.state, i, a, 1, true) - side_velocity<boundary_mode>(grid, grid.state, i, a, -1, true)) / spacing(grid.layout, i);
                float v = fmaxf(fabsf(side_velocity<boundary_mode>(grid, grid.state, i, a, -1)), fabsf(side_velocity<boundary_mode>(grid, grid.state, i, a, 1)));
                velocity_bound += full ? v * v : v;
            }
            if constexpr (full) divergence = fmax(divergence, double(fabsf(div)));
            speed        = fmax(speed, double(full ? sqrtf(velocity_bound) : velocity_bound));
            float direction = full ? sqrtf(config.up.x * config.up.x + config.up.y * config.up.y + config.up.z * config.up.z) : fabsf(config.up.x) + fabsf(config.up.y) + fabsf(config.up.z);
            acceleration = fmax(acceleration, double(fabsf(config.buoyancy * grid.state.temperature[i] - config.smoke_weight * grid.state.smoke[i]) * direction));
        }
        if constexpr (full) {
            double values[6]{fluid, divergence, speed, smoke, heat, acceleration};
            for (int k = 0; k < count; ++k) sums[k][threadIdx.x] = values[k];
        } else {
            sums[0][threadIdx.x] = speed;
            sums[1][threadIdx.x] = acceleration;
        }
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                for (int k = 0; k < count; ++k) sums[k][threadIdx.x] = full && (k == 0 || k == 3 || k == 4) ? sums[k][threadIdx.x] + sums[k][threadIdx.x + stride] : fmax(sums[k][threadIdx.x], sums[k][threadIdx.x + stride]);
            __syncthreads();
        }
        if (!threadIdx.x)
            for (int k = 0; k < count; ++k) results[b * count + k] = sums[k][0];
    }

    template <bool full>
    __global__ void finish_measure(Grid grid, const SourceView* sources, int source_count, float time, double* results) {
        constexpr int count = full ? 6 : 2;
        int k               = threadIdx.x;
        if (k >= count) return;
        double value = 0;
        for (int b = 0; b < grid.layout.count; ++b) value = full && (k == 0 || k == 3 || k == 4) ? value + results[b * count + k] : fmax(value, results[b * count + k]);
        if (k == (full ? 2 : 0)) {
            constexpr float motion_metric = full ? 1.f : 1.732050808f;
            for (int s = 0; s < source_count; ++s)
                if (time >= sources[s].start_time && time <= sources[s].end_time) {
                    float source_speed = full ? sqrtf(sources[s].velocity.x * sources[s].velocity.x + sources[s].velocity.y * sources[s].velocity.y + sources[s].velocity.z * sources[s].velocity.z) : fabsf(sources[s].velocity.x) + fabsf(sources[s].velocity.y) + fabsf(sources[s].velocity.z);
                    value = fmax(value, double(source_speed + motion_metric * sources[s].instance.max_speed * sources[s].inherit_velocity));
                }
            value = fmax(value, double(motion_metric * grid.collider_speed));
        }
        results[grid.layout.count * count + k] = value;
    }

    void measure(Grid grid, ForcesConfig config, const SourceView* sources, int source_count, float time, bool full, double* reductions, ::cuda::stream_ref stream) {
        if (!full) {
            if (grid.layout.count) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), measure_blocks<false, BoundaryMode::none>, grid, config, reductions);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<32>()), finish_measure<false>, grid, sources, source_count, time, reductions);
            return;
        }
        if (grid.layout.count) {
            switch (grid.boundary_mode) {
            case BoundaryMode::none: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), measure_blocks<true, BoundaryMode::none>, grid, config, reductions); break;
            case BoundaryMode::stationary: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), measure_blocks<true, BoundaryMode::stationary>, grid, config, reductions); break;
            case BoundaryMode::moving: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), measure_blocks<true, BoundaryMode::moving>, grid, config, reductions); break;
            }
        }
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<32>()), finish_measure<true>, grid, sources, source_count, time, reductions);
    }
}
