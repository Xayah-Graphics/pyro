#include "cuda.cuh"

namespace pyro::cuda::dense {
    __device__ float3 trace(Grid grid, Fields carrier, float3 p, float3 wind, float h, float end_time, float dt, bool obstacles, bool& clipped) {
        Layout layout = grid.layout;
        clipped       = false;
        if (obstacles) {
            float3 world{grid.origin.x + p.x * h, grid.origin.y + p.y * h, grid.origin.z + p.z * h};
            float distance = sample_boundary(grid, world, end_time, h);
            if (distance < 0.01f * h) {
                float e = 0.25f * h;
                float3 gradient{sample_boundary(grid, {world.x + e, world.y, world.z}, end_time, h) - sample_boundary(grid, {world.x - e, world.y, world.z}, end_time, h), sample_boundary(grid, {world.x, world.y + e, world.z}, end_time, h) - sample_boundary(grid, {world.x, world.y - e, world.z}, end_time, h), sample_boundary(grid, {world.x, world.y, world.z + e}, end_time, h) - sample_boundary(grid, {world.x, world.y, world.z - e}, end_time, h)};
                float scale = (0.01f * h - distance) / (h * fmaxf(sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z), 1.0e-20f));
                p           = {p.x + gradient.x * scale, p.y + gradient.y * scale, p.z + gradient.z * scale};
                clipped     = true;
            }
        }
        float3 first  = sample_velocity(layout, carrier, p, wind);
        float3 middle = sample_velocity(layout, carrier, {p.x - first.x * (0.5f * dt / h), p.y - first.y * (0.5f * dt / h), p.z - first.z * (0.5f * dt / h)}, wind);
        float3 departure{p.x - middle.x * (dt / h), p.y - middle.y * (dt / h), p.z - middle.z * (dt / h)};
        if (!obstacles) return departure;
        float3 delta{departure.x - p.x, departure.y - p.y, departure.z - p.z};
        float bound = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) * h + grid.collider_speed * fabsf(dt);
        if (bound < 1.0e-12f) return departure;
        float fraction = 0.0f;
        for (int step = 0; step < 64; ++step) {
            float3 position{p.x + delta.x * fraction, p.y + delta.y * fraction, p.z + delta.z * fraction};
            float distance = sample_boundary(grid, {grid.origin.x + position.x * h, grid.origin.y + position.y * h, grid.origin.z + position.z * h}, end_time - dt * fraction, h);
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

    template <bool correct>
    __global__ void transport_velocity(Grid grid, Config config, Fields destination, Fields original, Fields carrier, bool obstacles, float time, float dt) {
        int id = static_cast<int>(blockIdx.x), axis = blockIdx.y;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i         = id * tile_cells + local;
            float* output = destination.velocity[axis];
            if (!face_valid(grid.layout, cell_coordinate(grid.layout, i), axis)) {
                output[i] = (axis == 0 ? config.wind.x : axis == 1 ? config.wind.y : config.wind.z);
                continue;
            }
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
            float background   = (axis == 0 ? config.wind.x : axis == 1 ? config.wind.y : config.wind.z);
            bool clipped;
            float3 departure = trace(grid, carrier, p, config.wind, config.cell_size, time + dt, dt, obstacles, clipped);
            float lo, hi;
            float transported = sample(grid.layout, input, {departure.x - stagger.x, departure.y - stagger.y, departure.z - stagger.z}, background, &lo, &hi);
            if (correct && !clipped) {
                float3 reverse = trace(grid, carrier, p, config.wind, config.cell_size, time, -dt, obstacles, clipped);
                if (!clipped) transported = fminf(hi, fmaxf(lo, transported + 0.5f * (input[i] - sample(grid.layout, grid.transported.velocity[axis], {reverse.x - stagger.x, reverse.y - stagger.y, reverse.z - stagger.z}, background))));
            }
            output[i] = transported;
        }
    }

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
                    lo.x = fminf(lo.x, smoke);
                    lo.y = fminf(lo.y, heat);
                    hi.x = fmaxf(hi.x, smoke);
                    hi.y = fmaxf(hi.y, heat);
                }
        if (lower) *lower = lo;
        if (upper) *upper = hi;
        return result;
    }

    template <bool correct>
    __global__ void transport_scalars(Grid grid, Config config, Fields destination, Fields original, Fields carrier, bool obstacles, float time, float dt) {
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
            float3 departure = trace(grid, carrier, p, config.wind, config.cell_size, time + dt, dt, obstacles, clipped);
            float2 lo, hi;
            float2 transported = sample_scalars(grid.layout, original, {departure.x - stagger.x, departure.y - stagger.y, departure.z - stagger.z}, &lo, &hi);
            if (correct && !clipped) {
                float3 reverse = trace(grid, carrier, p, config.wind, config.cell_size, time, -dt, obstacles, clipped);
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

    __global__ void forces_kernel(Grid grid, Config config, Fields destination, Fields force_fields, float dt) {
        float smoke_decay = expf(-0.5f * config.dissipation * dt);
        float heat_decay  = expf(-0.5f * config.cooling * dt);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i                      = blockIdx.x * tile_cells + local;
            int3 c                     = cell_coordinate(grid.layout, i);
            destination.smoke[i]       = grid.fluid[i] ? destination.smoke[i] * smoke_decay : 0;
            destination.temperature[i] = grid.fluid[i] ? destination.temperature[i] * heat_decay : 0;
            for (int axis = 0; axis < 3; ++axis) {
                if (!face_valid(grid.layout, c, axis)) {
                    destination.velocity[axis][i] = (axis == 0 ? config.wind.x : axis == 1 ? config.wind.y : config.wind.z);
                    continue;
                }
                int previous      = neighbor(grid.layout, i, axis, -1);
                float density     = 0.5f * (force_fields.smoke[i] + value(force_fields.smoke, previous));
                float temperature = 0.5f * (force_fields.temperature[i] + value(force_fields.temperature, previous));
                destination.velocity[axis][i] += (axis == 0 ? config.up.x : axis == 1 ? config.up.y : config.up.z) * (config.buoyancy * temperature - config.smoke_weight * density) * dt;
            }
        }
    }

    __global__ void cache_source_kernel(Grid grid, Config config, const Source* sources, int index, SourceRegion region) {
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        int3 extent = region.extent;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        int3 c{region.lower.x + ordinal % extent.x, region.lower.y + ordinal / extent.x % extent.y, region.lower.z + ordinal / (extent.x * extent.y)};
        float3 p{grid.origin.x + (c.x + .5f) * config.cell_size, grid.origin.y + (c.y + .5f) * config.cell_size, grid.origin.z + (c.z + .5f) * config.cell_size};
        region.weights[ordinal * 4] = source_weight(sources[index], p, config.cell_size, sources[index].motion.time);
        for (int axis = 0; axis < 3; ++axis) {
            float3 face = p;
            if (axis == 0) face.x -= config.cell_size * .5f;
            if (axis == 1) face.y -= config.cell_size * .5f;
            if (axis == 2) face.z -= config.cell_size * .5f;
            region.weights[ordinal * 4 + axis + 1] = source_weight(sources[index], face, config.cell_size, sources[index].motion.time);
        }
    }

    template <bool cached>
    __global__ void source_kernel(Grid grid, Config config, Fields destination, const Source* sources, int source_index, SourceRegion region, float time, float dt) {
        int3 lower = region.lower, extent = region.extent;
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        Source source = sources[source_index];
        float overlap = fmaxf(0.0f, fminf(time + dt, source.end_time) - fmaxf(time, source.start_time));
        if (overlap == 0) return;
        int3 c{lower.x + ordinal % extent.x, lower.y + ordinal / extent.x % extent.y, lower.z + ordinal / (extent.x * extent.y)};
        int i = cell_index(grid.layout, c);
        float3 p{grid.origin.x + (c.x + 0.5f) * config.cell_size, grid.origin.y + (c.y + 0.5f) * config.cell_size, grid.origin.z + (c.z + 0.5f) * config.cell_size};
        if (grid.fluid[i]) {
            float weight = (cached ? region.weights[ordinal * 4] : source_weight(source, p, config.cell_size, time + dt * 0.5f)) * overlap;
            destination.smoke[i] += source.smoke_rate * weight;
            destination.temperature[i] += source.temperature_rate * weight;
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (!face_valid(grid.layout, c, axis)) continue;
            float3 face = p;
            if (axis == 0) face.x -= config.cell_size * 0.5f;
            if (axis == 1) face.y -= config.cell_size * 0.5f;
            if (axis == 2) face.z -= config.cell_size * 0.5f;
            float3 motion_velocity{};
            float weight                  = cached ? region.weights[ordinal * 4 + axis + 1] : source_weight(source, face, config.cell_size, time + dt * 0.5f, &motion_velocity);
            float blend                   = -expm1f(-source.velocity_rate * weight * overlap);
            float velocity                = destination.velocity[axis][i];
            float target                  = axis == 0 ? source.velocity.x + motion_velocity.x * source.inherit_velocity : axis == 1 ? source.velocity.y + motion_velocity.y * source.inherit_velocity : source.velocity.z + motion_velocity.z * source.inherit_velocity;
            destination.velocity[axis][i] = velocity + blend * (target - velocity);
        }
    }

    __global__ void decay_kernel(Config config, Fields destination, float dt) {
        float smoke_decay = expf(-0.5f * config.dissipation * dt);
        float heat_decay  = expf(-0.5f * config.cooling * dt);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            destination.smoke[i] *= smoke_decay;
            destination.temperature[i] *= heat_decay;
        }
    }

    template <bool full>
    __global__ void statistics_kernel(Grid grid, Fields fields, Config config, double* partial) {
        __shared__ double sums[6][threads];
        double values[6]{};
        int id       = static_cast<int>(blockIdx.x);
        float volume = config.cell_size * config.cell_size * config.cell_size;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i       = id * tile_cells + local;
            float speed = sqrtf(fields.velocity[0][i] * fields.velocity[0][i] + fields.velocity[1][i] * fields.velocity[1][i] + fields.velocity[2][i] * fields.velocity[2][i]);
            values[2]   = fmax(values[2], static_cast<double>(isfinite(speed) ? speed : INFINITY));
            if (!grid.fluid[i]) continue;
            if constexpr (full) {
                float divergence = 0.0f;
                for (int axis = 0; axis < 3; ++axis) {
                    int right = neighbor(grid.layout, i, axis, 1);
                    float a = grid.aperture[axis][i], b = value(grid.aperture[axis], right, 1.0f);
                    divergence += b * value(fields.velocity[axis], right, (axis == 0 ? config.wind.x : axis == 1 ? config.wind.y : config.wind.z)) + (1.0f - b) * value(grid.solid_velocity[axis], right) - a * fields.velocity[axis][i] - (1.0f - a) * grid.solid_velocity[axis][i];
                }
                divergence = divergence / config.cell_size - grid.solid_divergence[i];
                values[0] += 1.0;
                values[1] = fmax(values[1], static_cast<double>(isfinite(divergence) ? fabsf(divergence) : INFINITY));
                values[3] += fields.smoke[i] * volume;
                values[4] += fields.temperature[i] * volume;
            }
            values[5] = fmax(values[5], static_cast<double>(fabsf(config.buoyancy * fields.temperature[i] - config.smoke_weight * fields.smoke[i]) * sqrtf(config.up.x * config.up.x + config.up.y * config.up.y + config.up.z * config.up.z)));
        }
        for (int k = 0; k < 6; ++k) sums[k][threadIdx.x] = values[k];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                for (int k = 0; k < 6; ++k) {
                    if (k == 1 || k == 2 || k == 5) sums[k][threadIdx.x] = fmax(sums[k][threadIdx.x], sums[k][threadIdx.x + stride]);
                    else sums[k][threadIdx.x] += sums[k][threadIdx.x + stride];
                }
            __syncthreads();
        }
        if (!threadIdx.x)
            for (int k = 0; k < 6; ++k) partial[6 * blockIdx.x + k] = sums[k][0];
    }

    __global__ void finish_statistics(int count, Config config, const Source* sources, int source_count, float collider_speed, float time, double* partial, float* maximum) {
        __shared__ double sums[6][threads];
        double values[6]{};
        for (int i = threadIdx.x; i < count; i += blockDim.x)
            for (int k = 0; k < 6; ++k) {
                if (k == 1 || k == 2 || k == 5) values[k] = fmax(values[k], partial[6 * i + k]);
                else values[k] += partial[6 * i + k];
            }
        for (int k = 0; k < 6; ++k) sums[k][threadIdx.x] = values[k];
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                for (int k = 0; k < 6; ++k) {
                    if (k == 1 || k == 2 || k == 5) sums[k][threadIdx.x] = fmax(sums[k][threadIdx.x], sums[k][threadIdx.x + stride]);
                    else sums[k][threadIdx.x] += sums[k][threadIdx.x + stride];
                }
            __syncthreads();
        }
        if (threadIdx.x) return;
        for (int k = 0; k < 6; ++k) values[k] = sums[k][0];
        for (int k = 0; k < 6; ++k) partial[count * 6ull + k] = values[k];
        float speed = fmaxf(static_cast<float>(values[2]), sqrtf(config.wind.x * config.wind.x + config.wind.y * config.wind.y + config.wind.z * config.wind.z)), acceleration = static_cast<float>(values[5]);
        for (int i = 0; i < source_count; ++i) {
            Source source = sources[i];
            if (time >= source.end_time || time + config.max_time_step < source.start_time) continue;
            speed = fmaxf(speed, sqrtf(source.velocity.x * source.velocity.x + source.velocity.y * source.velocity.y + source.velocity.z * source.velocity.z) + source.motion.max_speed * fmaxf(1.0f, source.inherit_velocity));
            acceleration += (fabsf(config.buoyancy * source.temperature_rate) + fabsf(config.smoke_weight * source.smoke_rate)) * config.max_time_step * sqrtf(config.up.x * config.up.x + config.up.y * config.up.y + config.up.z * config.up.z);
        }
        speed      = fmaxf(speed, collider_speed);
        maximum[0] = speed;
        maximum[1] = acceleration;
    }

    void advect(Grid grid, Config config, Fields destination, Fields original, Fields carrier, bool obstacles, float start_time, float dt, cudaStream_t stream) {
        dim3 velocity_blocks(grid.layout.blocks, 3);
        transport_velocity<false><<<velocity_blocks, threads, 0, stream>>>(grid, config, grid.transported, original, carrier, obstacles, start_time, dt);
        transport_scalars<false><<<grid.layout.blocks, threads, 0, stream>>>(grid, config, grid.transported, original, carrier, obstacles, start_time, dt);
        transport_velocity<true><<<velocity_blocks, threads, 0, stream>>>(grid, config, destination, original, carrier, obstacles, start_time, dt);
        transport_scalars<true><<<grid.layout.blocks, threads, 0, stream>>>(grid, config, destination, original, carrier, obstacles, start_time, dt);
    }

    void cache_source(Grid grid, Config config, const Source* sources, int index, SourceRegion region, cudaStream_t stream) {
        int cells = region.extent.x * region.extent.y * region.extent.z;
        if (cells) cache_source_kernel<<<(cells + threads - 1) / threads, threads, 0, stream>>>(grid, config, sources, index, region);
    }

    void apply_sources_forces(Grid grid, Config config, Fields destination, Fields force_fields, const Source* sources, const SourceRegion* regions, int count, float time, float dt, cudaStream_t stream) {
        forces_kernel<<<grid.layout.blocks, threads, 0, stream>>>(grid, config, destination, force_fields, dt);
        for (int source = 0; source < count; ++source) {
            auto region = regions[source];
            int cells   = region.extent.x * region.extent.y * region.extent.z;
            if (!cells) continue;
            if (region.cached) source_kernel<true><<<(cells + threads - 1) / threads, threads, 0, stream>>>(grid, config, destination, sources, source, region, time, dt);
            else source_kernel<false><<<(cells + threads - 1) / threads, threads, 0, stream>>>(grid, config, destination, sources, source, region, time, dt);
        }
        decay_kernel<<<grid.layout.blocks, threads, 0, stream>>>(config, destination, dt);
    }

    void measure(Grid grid, Fields fields, Config config, const Source* sources, int source_count, float time, double* reductions, float* maximum, bool full, cudaStream_t stream) {
        if (full) statistics_kernel<true><<<grid.layout.blocks, threads, 0, stream>>>(grid, fields, config, reductions);
        else statistics_kernel<false><<<grid.layout.blocks, threads, 0, stream>>>(grid, fields, config, reductions);
        finish_statistics<<<1, threads, 0, stream>>>(grid.layout.blocks, config, sources, source_count, grid.collider_speed, time, reductions, maximum);
    }
} // namespace pyro::cuda::dense
