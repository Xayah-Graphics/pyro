#include "cuda.cuh"

namespace pyro::cuda {
    __device__ float3 trace(Grid grid, Fields carrier, float3 p, float3 wind, float h, float endTime, float dt, const Collider* colliders, int count, bool& clipped) {
        Layout layout = grid.layout;
        clipped       = false;
        if (count) {
            float3 world   = p * h;
            float distance = sample_boundary(grid, world, endTime, h);
            if (distance < 0.01f * h) {
                // A partially open face may have its MAC center inside the
                // solid. Start its characteristic on the fluid side instead
                // of freezing this sample at its own previous velocity.
                float e = 0.25f * h;
                float3 gradient{sample_boundary(grid, world + float3{e, 0, 0}, endTime, h) - sample_boundary(grid, world - float3{e, 0, 0}, endTime, h), sample_boundary(grid, world + float3{0, e, 0}, endTime, h) - sample_boundary(grid, world - float3{0, e, 0}, endTime, h), sample_boundary(grid, world + float3{0, 0, e}, endTime, h) - sample_boundary(grid, world - float3{0, 0, e}, endTime, h)};
                p       = p + gradient * ((0.01f * h - distance) / (h * fmaxf(length(gradient), 1.0e-20f)));
                clipped = true;
            }
        }
        float3 first     = sample_velocity(layout, carrier, p, wind);
        float3 middle    = sample_velocity(layout, carrier, p - first * (0.5f * dt / h), wind);
        float3 departure = p - middle * (dt / h);
        if (!count) return departure;
        float motion = 0.0f;
        for (int i = 0; i < count; ++i) motion = fmaxf(motion, (colliders[i].shape == Shape::mesh ? colliders[i].motion.maxSpeed : length(colliders[i].linearVelocity) + length(colliders[i].angularVelocity) * length(colliders[i].halfExtent)));
        float bound = length(departure - p) * h + motion * fabsf(dt);
        if (bound < 1.0e-12f) return departure;
        float fraction = 0.0f;
        // Conservative advancement through the swept space-time SDF.
        for (int step = 0; step < 64; ++step) {
            float3 position = p + (departure - p) * fraction;
            float distance  = sample_boundary(grid, position * h, endTime - dt * fraction, h);
            if (distance < 0.001f * h) {
                clipped = true;
                return position;
            }
            fraction += 0.5f * distance / bound;
            if (fraction >= 1.0f) return departure;
        }
        clipped = true;
        return p + (departure - p) * fraction;
    }

    template <bool correct>
    __global__ void transport_velocity(Grid grid, Config config, Fields destination, Fields original, Fields carrier, const Collider* colliders, int count, float time, float dt) {
        int id = grid.layout.active[blockIdx.x], axis = blockIdx.y;
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i         = id * blockCells + local;
            float* output = destination.velocity[axis];
            if (grid.aperture[axis][i] == 0.0f) {
                output[i] = grid.solidVelocity[axis][i];
                continue;
            }
            int3 c = cell_coordinate(grid.layout, i);
            float3 stagger{0.5f, 0.5f, 0.5f};
            if (axis == 0) stagger.x = 0.0f;
            if (axis == 1) stagger.y = 0.0f;
            if (axis == 2) stagger.z = 0.0f;
            float3 p{c.x + stagger.x, c.y + stagger.y, c.z + stagger.z};
            const float* input = original.velocity[axis];
            float background   = component(config.wind, axis);
            bool clipped;
            float3 departure = trace(grid, carrier, p, config.wind, config.cellSize, time + dt, dt, colliders, count, clipped);
            float lo, hi;
            float transported = sample(grid.layout, input, departure - stagger, background, &lo, &hi);
            if (correct && !clipped) {
                float3 reverse = trace(grid, carrier, p, config.wind, config.cellSize, time, -dt, colliders, count, clipped);
                if (!clipped) transported = fminf(hi, fmaxf(lo, transported + 0.5f * (input[i] - sample(grid.layout, grid.transported.velocity[axis], reverse - stagger, background))));
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
    __global__ void transport_scalars(Grid grid, Config config, Fields destination, Fields original, Fields carrier, const Collider* colliders, int count, float time, float dt) {
        int id = grid.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            if (!grid.fluid[i]) {
                destination.smoke[i]       = 0.0f;
                destination.temperature[i] = 0.0f;
                continue;
            }
            int3 c = cell_coordinate(grid.layout, i);
            float3 stagger{0.5f, 0.5f, 0.5f}, p{c.x + 0.5f, c.y + 0.5f, c.z + 0.5f};
            bool clipped;
            float3 departure = trace(grid, carrier, p, config.wind, config.cellSize, time + dt, dt, colliders, count, clipped);
            float2 lo, hi;
            float2 transported = sample_scalars(grid.layout, original, departure - stagger, &lo, &hi);
            if (correct && !clipped) {
                float3 reverse = trace(grid, carrier, p, config.wind, config.cellSize, time, -dt, colliders, count, clipped);
                if (!clipped) {
                    float2 backward = sample_scalars(grid.layout, grid.transported, reverse - stagger);
                    transported.x   = fminf(hi.x, fmaxf(lo.x, transported.x + 0.5f * (original.smoke[i] - backward.x)));
                    transported.y   = fminf(hi.y, fmaxf(lo.y, transported.y + 0.5f * (original.temperature[i] - backward.y)));
                }
            }
            destination.smoke[i]       = transported.x;
            destination.temperature[i] = transported.y;
        }
    }

    __global__ void forces_kernel(Grid grid, Config config, Fields destination, Fields forceFields, const Source* sources, int count, float time, float dt) {
        int id = grid.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i  = id * blockCells + local;
            int3 c = cell_coordinate(grid.layout, i);
            float3 p{(c.x + 0.5f) * config.cellSize, (c.y + 0.5f) * config.cellSize, (c.z + 0.5f) * config.cellSize};
            // Split decay symmetrically around source injection.
            float smokeDecay = expf(-0.5f * config.dissipation * dt);
            float heatDecay  = expf(-0.5f * config.cooling * dt);
            float smoke = destination.smoke[i] * smokeDecay, heat = destination.temperature[i] * heatDecay;
            for (int s = 0; s < count; ++s) {
                Source source = sources[s];
                float overlap = fmaxf(0.0f, fminf(time + dt, source.endTime) - fmaxf(time, source.startTime));
                float weight  = source_weight(source, p, config.cellSize, time + dt * 0.5f) * overlap;
                smoke += source.smokeRate * weight;
                heat += source.temperatureRate * weight;
            }
            destination.smoke[i]       = grid.fluid[i] ? smoke * smokeDecay : 0.0f;
            destination.temperature[i] = grid.fluid[i] ? heat * heatDecay : 0.0f;
            for (int axis = 0; axis < 3; ++axis) {
                int previous      = neighbor(grid.layout, i, axis, -1);
                float density     = 0.5f * (forceFields.smoke[i] + value(forceFields.smoke, previous));
                float temperature = 0.5f * (forceFields.temperature[i] + value(forceFields.temperature, previous));
                float velocity    = destination.velocity[axis][i] + component(config.up, axis) * (config.buoyancy * temperature - config.smokeWeight * density) * dt;
                float3 face       = p;
                if (axis == 0) face.x -= config.cellSize * 0.5f;
                if (axis == 1) face.y -= config.cellSize * 0.5f;
                if (axis == 2) face.z -= config.cellSize * 0.5f;
                for (int s = 0; s < count; ++s) {
                    Source source = sources[s];
                    float overlap = fmaxf(0.0f, fminf(time + dt, source.endTime) - fmaxf(time, source.startTime));
                    float3 motionVelocity{};
                    float blend = -expm1f(-source.velocityRate * source_weight(source, face, config.cellSize, time + dt * 0.5f, &motionVelocity) * overlap);
                    velocity += blend * (component(source.velocity + motionVelocity * source.inheritVelocity, axis) - velocity);
                }
                destination.velocity[axis][i] = velocity;
            }
        }
    }

    template <bool full>
    __global__ void statistics_kernel(Grid grid, Fields fields, Config config, double* partial) {
        __shared__ double sums[6][threads];
        double values[6]{};
        int id       = grid.layout.active[blockIdx.x];
        float volume = config.cellSize * config.cellSize * config.cellSize;
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i       = id * blockCells + local;
            float speed = length({fields.velocity[0][i], fields.velocity[1][i], fields.velocity[2][i]});
            values[2]   = fmax(values[2], static_cast<double>(isfinite(speed) ? speed : INFINITY));
            if (!grid.fluid[i]) continue;
            if constexpr (full) {
                float divergence = 0.0f;
                for (int axis = 0; axis < 3; ++axis) {
                    int right = neighbor(grid.layout, i, axis, 1);
                    float a = grid.aperture[axis][i], b = value(grid.aperture[axis], right, 1.0f);
                    divergence += b * value(fields.velocity[axis], right, component(config.wind, axis)) + (1.0f - b) * value(grid.solidVelocity[axis], right) - a * fields.velocity[axis][i] - (1.0f - a) * grid.solidVelocity[axis][i];
                }
                divergence = divergence / config.cellSize - grid.solidDivergence[i];
                values[0] += 1.0;
                values[1] = fmax(values[1], static_cast<double>(isfinite(divergence) ? fabsf(divergence) : INFINITY));
                values[3] += fields.smoke[i] * volume;
                values[4] += fields.temperature[i] * volume;
            }
            values[5] = fmax(values[5], static_cast<double>(fabsf(config.buoyancy * fields.temperature[i] - config.smokeWeight * fields.smoke[i]) * length(config.up)));
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

    __global__ void finish_statistics(int count, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, double* partial, float* maximum) {
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
        for (int k = 0; k < 6; ++k) partial[config.maxBlocks * 6ull + k] = values[k];
        float speed = fmaxf(static_cast<float>(values[2]), length(config.wind)), acceleration = static_cast<float>(values[5]);
        for (int i = 0; i < sourceCount; ++i) {
            Source source = sources[i];
            if (time >= source.endTime || time + config.maxTimeStep < source.startTime) continue;
            speed = fmaxf(speed, length(source.velocity) + source.motion.maxSpeed * fmaxf(1.0f, source.inheritVelocity));
            acceleration += (fabsf(config.buoyancy * source.temperatureRate) + fabsf(config.smokeWeight * source.smokeRate)) * config.maxTimeStep * length(config.up);
        }
        for (int i = 0; i < colliderCount; ++i) speed = fmaxf(speed, (colliders[i].shape == Shape::mesh ? colliders[i].motion.maxSpeed : length(colliders[i].linearVelocity) + length(colliders[i].angularVelocity) * length(colliders[i].halfExtent)));
        maximum[0] = speed;
        maximum[1] = acceleration;
    }

    void advect(Grid grid, Config config, Fields destination, Fields original, Fields carrier, const Collider* colliders, int count, float startTime, float dt, cudaStream_t stream) {
        dim3 velocityBlocks(grid.layout.blocks, 3);
        transport_velocity<false><<<velocityBlocks, threads, 0, stream>>>(grid, config, grid.transported, original, carrier, colliders, count, startTime, dt);
        transport_scalars<false><<<grid.layout.blocks, threads, 0, stream>>>(grid, config, grid.transported, original, carrier, colliders, count, startTime, dt);
        transport_velocity<true><<<velocityBlocks, threads, 0, stream>>>(grid, config, destination, original, carrier, colliders, count, startTime, dt);
        transport_scalars<true><<<grid.layout.blocks, threads, 0, stream>>>(grid, config, destination, original, carrier, colliders, count, startTime, dt);
    }

    void apply_sources_forces(Grid grid, Config config, Fields destination, Fields forceFields, const Source* sources, int count, float time, float dt, cudaStream_t stream) {
        forces_kernel<<<grid.layout.blocks, threads, 0, stream>>>(grid, config, destination, forceFields, sources, count, time, dt);
    }

    void measure(Grid grid, Fields fields, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, double* reductions, float* maximum, bool full, cudaStream_t stream) {
        if (grid.layout.blocks) {
            if (full) statistics_kernel<true><<<grid.layout.blocks, threads, 0, stream>>>(grid, fields, config, reductions);
            else statistics_kernel<false><<<grid.layout.blocks, threads, 0, stream>>>(grid, fields, config, reductions);
        }
        finish_statistics<<<1, threads, 0, stream>>>(grid.layout.blocks, config, sources, sourceCount, colliders, colliderCount, time, reductions, maximum);
    }
} // namespace pyro::cuda
