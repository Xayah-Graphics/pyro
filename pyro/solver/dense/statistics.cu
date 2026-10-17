#include "statistics.cuh"
#include <cuda/launch>
#include <solver/common/scene.cuh>

namespace pyro::dense {
    template <bool full, BoundaryMode boundary_mode>
    __global__ void statistics_kernel(Grid grid, Fields fields, ForcesConfig config, float cell_size, float3 wind, double* partial) {
        constexpr int channels = full ? 6 : 2;
        __shared__ double sums[channels][threads];
        double values[channels]{};
        int id       = static_cast<int>(blockIdx.x);
        float volume = cell_size * cell_size * cell_size;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i                = id * tile_cells + local;
            float speed = 0;
            if constexpr (full) speed = sqrtf(fields.velocity[0][i] * fields.velocity[0][i] + fields.velocity[1][i] * fields.velocity[1][i] + fields.velocity[2][i] * fields.velocity[2][i]);
            else {
                if (!grid.fluid[i]) continue;
                for (int axis = 0; axis < 3; ++axis) speed += fmaxf(fabsf(fields.velocity[axis][i]), fabsf(value(fields.velocity[axis], neighbor(grid.layout, i, axis, 1))));
            }
            values[full ? 2 : 0] = fmax(values[full ? 2 : 0], static_cast<double>(isfinite(speed) ? speed : INFINITY));
            if (!grid.fluid[i]) continue;
            if constexpr (full) {
                float divergence = 0.0f;
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) {
                    int right = neighbor(grid.layout, i, axis, 1);
                    float a = boundary_mode == BoundaryMode::none ? 1.f : grid.aperture[axis][i], b = boundary_mode == BoundaryMode::none ? 1.f : value(grid.aperture[axis], right, 1.0f);
                    divergence += b * value(fields.velocity[axis], right, (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z)) + (1.0f - b) * (boundary_mode == BoundaryMode::moving ? value(grid.solid_velocity[axis], right) : 0) - a * fields.velocity[axis][i] - (1.0f - a) * (boundary_mode == BoundaryMode::moving ? grid.solid_velocity[axis][i] : 0);
                }
                divergence = divergence / cell_size - (boundary_mode == BoundaryMode::moving ? grid.solid_divergence[i] : 0);
                values[0] += 1.0;
                values[1] = fmax(values[1], static_cast<double>(isfinite(divergence) ? fabsf(divergence) : INFINITY));
                values[3] += fields.smoke[i] * volume;
                values[4] += fields.temperature[i] * volume;
            }
            float direction = full ? sqrtf(config.up.x * config.up.x + config.up.y * config.up.y + config.up.z * config.up.z) : fabsf(config.up.x) + fabsf(config.up.y) + fabsf(config.up.z);
            values[full ? 5 : 1] = fmax(values[full ? 5 : 1], static_cast<double>(fabsf(config.buoyancy * fields.temperature[i] - config.smoke_weight * fields.smoke[i]) * direction));
        }
#pragma unroll
        for (int k = 0; k < channels; ++k) sums[k][threadIdx.x] = values[k];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride)
#pragma unroll
                for (int k = 0; k < channels; ++k) {
                    if (!full || k == 1 || k == 2 || k == 5) sums[k][threadIdx.x] = fmax(sums[k][threadIdx.x], sums[k][threadIdx.x + stride]);
                    else sums[k][threadIdx.x] += sums[k][threadIdx.x + stride];
                }
            __syncthreads();
        }
        if (!threadIdx.x)
#pragma unroll
            for (int k = 0; k < channels; ++k) partial[channels * blockIdx.x + k] = sums[k][0];
    }

    template <bool full>
    __global__ void finish_statistics(int count, ForcesConfig config, float3 wind, float max_time_step, const SourceView* sources, int source_count, float collider_speed, float time, double* partial, float* maximum) {
        constexpr int channels = full ? 6 : 2;
        __shared__ double sums[channels][threads];
        double values[channels]{};
        for (int i = threadIdx.x; i < count; i += blockDim.x)
#pragma unroll
            for (int k = 0; k < channels; ++k) {
                if (!full || k == 1 || k == 2 || k == 5) values[k] = fmax(values[k], partial[channels * i + k]);
                else values[k] += partial[channels * i + k];
            }
#pragma unroll
        for (int k = 0; k < channels; ++k) sums[k][threadIdx.x] = values[k];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride)
#pragma unroll
                for (int k = 0; k < channels; ++k) {
                    if (!full || k == 1 || k == 2 || k == 5) sums[k][threadIdx.x] = fmax(sums[k][threadIdx.x], sums[k][threadIdx.x + stride]);
                    else sums[k][threadIdx.x] += sums[k][threadIdx.x + stride];
                }
            __syncthreads();
        }
        if (threadIdx.x) return;
#pragma unroll
        for (int k = 0; k < channels; ++k) values[k] = sums[k][0];
#pragma unroll
        for (int k = 0; k < channels; ++k) partial[count * static_cast<unsigned long long>(channels) + k] = values[k];
        float speed = fmaxf(static_cast<float>(values[full ? 2 : 0]), full ? sqrtf(wind.x * wind.x + wind.y * wind.y + wind.z * wind.z) : fabsf(wind.x) + fabsf(wind.y) + fabsf(wind.z)), acceleration = static_cast<float>(values[full ? 5 : 1]);
        constexpr float motion_metric = full ? 1.f : 1.732050808f;
        for (int i = 0; i < source_count; ++i) {
            SourceView source = sources[i];
            if (time >= source.end_time || time + max_time_step < source.start_time) continue;
            float source_speed = full ? sqrtf(source.velocity.x * source.velocity.x + source.velocity.y * source.velocity.y + source.velocity.z * source.velocity.z) : fabsf(source.velocity.x) + fabsf(source.velocity.y) + fabsf(source.velocity.z);
            speed = fmaxf(speed, source_speed + motion_metric * source.instance.max_speed * fmaxf(1.0f, source.inherit_velocity));
            float direction = full ? sqrtf(config.up.x * config.up.x + config.up.y * config.up.y + config.up.z * config.up.z) : fabsf(config.up.x) + fabsf(config.up.y) + fabsf(config.up.z);
            acceleration += (fabsf(config.buoyancy * source.temperature_rate) + fabsf(config.smoke_weight * source.smoke_rate)) * max_time_step * direction;
        }
        speed      = fmaxf(speed, motion_metric * collider_speed);
        maximum[0] = speed;
        maximum[1] = acceleration;
    }

    void measure(Grid grid, Fields fields, ForcesConfig config, float cell_size, float3 wind, float max_time_step, const SourceView* sources, int source_count, float time, double* reductions, float* maximum, bool full, ::cuda::stream_ref stream) {
        if (!full) {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), statistics_kernel<false, BoundaryMode::none>, grid, fields, config, cell_size, wind, reductions);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_statistics<false>, grid.layout.blocks, config, wind, max_time_step, sources, source_count, grid.collider_speed, time, reductions, maximum);
            return;
        }
        switch (grid.boundary_mode) {
        case BoundaryMode::none: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), statistics_kernel<true, BoundaryMode::none>, grid, fields, config, cell_size, wind, reductions); break;
        case BoundaryMode::stationary: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), statistics_kernel<true, BoundaryMode::stationary>, grid, fields, config, cell_size, wind, reductions); break;
        case BoundaryMode::moving: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), statistics_kernel<true, BoundaryMode::moving>, grid, fields, config, cell_size, wind, reductions); break;
        }
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_statistics<true>, grid.layout.blocks, config, wind, max_time_step, sources, source_count, grid.collider_speed, time, reductions, maximum);
    }
}
