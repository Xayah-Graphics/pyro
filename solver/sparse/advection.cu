#include "cuda.cuh"

namespace pyro::cuda::sparse {
    template <bool obstacles>
    __device__ float3 trace(Grid grid, Fields carrier, Config config, float3 p, float time, float dt, bool& clipped) {
        clipped = false;
        float h = config.cell_size;
        if constexpr (obstacles) {
            float distance = sample_boundary(grid, p, time);
            if (distance < .01f * h) {
                float3 gradient{};
                float dx    = sample_boundary(grid, {p.x + h * .25f, p.y, p.z}, time) - sample_boundary(grid, {p.x - h * .25f, p.y, p.z}, time);
                float dy    = sample_boundary(grid, {p.x, p.y + h * .25f, p.z}, time) - sample_boundary(grid, {p.x, p.y - h * .25f, p.z}, time);
                float dz    = sample_boundary(grid, {p.x, p.y, p.z + h * .25f}, time) - sample_boundary(grid, {p.x, p.y, p.z - h * .25f}, time);
                gradient    = {dx, dy, dz};
                float scale = (.01f * h - distance) / fmaxf(sqrtf(gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z), 1.e-20f);
                p           = {p.x + gradient.x * scale, p.y + gradient.y * scale, p.z + gradient.z * scale};
                clipped     = true;
            }
        }
        float3 v = sample_velocity(grid, carrier, p, config.wind), middle = sample_velocity(grid, carrier, {p.x - v.x * (.5f * dt), p.y - v.y * (.5f * dt), p.z - v.z * (.5f * dt)}, config.wind);
        float3 departure{p.x - middle.x * dt, p.y - middle.y * dt, p.z - middle.z * dt};
        if constexpr (!obstacles) return departure;
        float3 delta{departure.x - p.x, departure.y - p.y, departure.z - p.z};
        float bound = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) + grid.collider_speed * fabsf(dt), fraction = 0;
        if (bound < 1.e-12f) return departure;
        for (int step = 0; step < 64; ++step) {
            float3 q{p.x + delta.x * fraction, p.y + delta.y * fraction, p.z + delta.z * fraction};
            float d = sample_boundary(grid, q, time - dt * fraction);
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
    template <bool correct, bool obstacles>
    __global__ void transport_cells(Grid grid, Fields destination, Fields carrier, Config config, float time, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        if (!grid.fluid[i]) {
            destination.smoke[i] = destination.temperature[i] = 0;
            return;
        }
        float3 p = center(grid.layout, i);
        bool clipped, reverse_clipped = true;
        float3 departure = trace<obstacles>(grid, carrier, config, p, time + dt, dt, clipped), reverse{};
        if (correct && !clipped) reverse = trace<obstacles>(grid, carrier, config, p, time, -dt, reverse_clipped);
        for (int field = 3; field < 5; ++field) {
            float lo, hi, value = sample(grid, grid.state, departure, field, 0, &lo, &hi);
            if (correct && !clipped && !reverse_clipped) value = fminf(hi, fmaxf(lo, value + .5f * ((field == 3 ? grid.state.smoke[i] : grid.state.temperature[i]) - sample(grid, grid.transported, reverse, field))));
            (field == 3 ? destination.smoke : destination.temperature)[i] = value;
        }
    }
    template <bool correct, bool obstacles>
    __global__ void transport_faces(Grid grid, Fields destination, Fields carrier, Config config, float time, float dt) {
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
        if (face.area <= 0) {
            destination.velocity[f] = face.solid_flux / (h * h);
            return;
        }
        bool clipped;
        float3 departure = trace<obstacles>(grid, carrier, config, p, time + dt, dt, clipped);
        float lo, hi, value = sample(grid, grid.state, departure, axis, (axis == 0 ? config.wind.x : axis == 1 ? config.wind.y : config.wind.z), &lo, &hi);
        if (correct && !clipped) {
            float3 reverse = trace<obstacles>(grid, carrier, config, p, time, -dt, clipped);
            if (!clipped) value = fminf(hi, fmaxf(lo, value + .5f * (grid.state.velocity[f] - sample(grid, grid.transported, reverse, axis, (axis == 0 ? config.wind.x : axis == 1 ? config.wind.y : config.wind.z)))));
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
    void advect(Grid grid, Fields destination, Fields carrier, Config config, bool obstacles, float time, float dt, cudaStream_t stream) {
        grid.state.samples       = grid.samples[0];
        grid.transported.samples = grid.samples[2];
        cache_samples<<<dim3(grid.layout.count, 5), 256, 0, stream>>>(grid, grid.state, config.wind, grid.samples[0]);
        if (carrier.velocity == grid.state.velocity) carrier.samples = grid.samples[0];
        else {
            carrier.samples = grid.samples[1];
            cache_samples<<<dim3(grid.layout.count, 5), 256, 0, stream>>>(grid, carrier, config.wind, grid.samples[1]);
        }

        if (obstacles) {
            transport_cells<false, true><<<grid.layout.count * 2, 256, 0, stream>>>(grid, grid.transported, carrier, config, time, dt);
            transport_faces<false, true><<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, grid.transported, carrier, config, time, dt);
            cache_samples<<<dim3(grid.layout.count, 5), 256, 0, stream>>>(grid, grid.transported, config.wind, grid.samples[2]);
            transport_cells<true, true><<<grid.layout.count * 2, 256, 0, stream>>>(grid, destination, carrier, config, time, dt);
            transport_faces<true, true><<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, destination, carrier, config, time, dt);
        } else {
            transport_cells<false, false><<<grid.layout.count * 2, 256, 0, stream>>>(grid, grid.transported, carrier, config, time, dt);
            transport_faces<false, false><<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, grid.transported, carrier, config, time, dt);
            cache_samples<<<dim3(grid.layout.count, 5), 256, 0, stream>>>(grid, grid.transported, config.wind, grid.samples[2]);
            transport_cells<true, false><<<grid.layout.count * 2, 256, 0, stream>>>(grid, destination, carrier, config, time, dt);
            transport_faces<true, false><<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, destination, carrier, config, time, dt);
        }
    }
    __global__ void scalar_forces(Grid grid, Fields destination, Config config, const Source* sources, int count, float time, float dt) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        float smoke = destination.smoke[i] * expf(-.5f * config.dissipation * dt), temperature = destination.temperature[i] * expf(-.5f * config.cooling * dt);
        for (int s = 0; s < count; ++s) {
            Source source    = sources[s];
            source.mesh.band = .5f * spacing(grid.layout, i);
            float active     = fmaxf(0, fminf(time + dt, source.end_time) - fmaxf(time, source.start_time));
            if (!active) continue;
            float weight = source_weight(source, center(grid.layout, i), spacing(grid.layout, i), time + dt * .5f);
            smoke += source.smoke_rate * weight * active;
            temperature += source.temperature_rate * weight * active;
        }
        destination.smoke[i]       = grid.fluid[i] ? smoke * expf(-.5f * config.dissipation * dt) : 0;
        destination.temperature[i] = grid.fluid[i] ? temperature * expf(-.5f * config.cooling * dt) : 0;
    }
    __global__ void velocity_forces(Grid grid, Fields destination, Fields carrier, Config config, const Source* sources, int count, float time, float dt) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728 || grid.faces[f].area <= 0) return;
        Face face = grid.faces[f];
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        float smoke       = .5f * ((face.left >= 0 ? carrier.smoke[face.left] : 0) + (face.right >= 0 ? carrier.smoke[face.right] : 0));
        float temperature = .5f * ((face.left >= 0 ? carrier.temperature[face.left] : 0) + (face.right >= 0 ? carrier.temperature[face.right] : 0));
        float v           = destination.velocity[f] + (axis == 0 ? config.up.x : axis == 1 ? config.up.y : config.up.z) * (config.buoyancy * temperature - config.smoke_weight * smoke) * dt;
        for (int s = 0; s < count; ++s) {
            Source source    = sources[s];
            source.mesh.band = .5f * h;
            float active     = fmaxf(0, fminf(time + dt, source.end_time) - fmaxf(time, source.start_time));
            if (!active) continue;
            float3 motion{};
            float weight = source_weight(source, p, h, time + dt * .5f, &motion);
            float target = axis == 0 ? source.velocity.x + motion.x * source.inherit_velocity : axis == 1 ? source.velocity.y + motion.y * source.inherit_velocity : source.velocity.z + motion.z * source.inherit_velocity;
            v += (target - v) * (1 - expf(-source.velocity_rate * weight * active));
        }
        destination.velocity[f] = v;
    }
    void forces(Grid grid, Fields destination, Fields carrier, Config config, const Source* sources, int count, float time, float dt, cudaStream_t stream) {
        scalar_forces<<<grid.layout.count * 2, 256, 0, stream>>>(grid, destination, config, sources, count, time, dt);
        velocity_forces<<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, destination, carrier, config, sources, count, time, dt);
    }
    template <bool full>
    __global__ void measure_blocks(Grid grid, Config config, double* results) {
        constexpr int count = full ? 6 : 2;
        __shared__ double sums[count][256];
        int b        = blockIdx.x;
        double fluid = 0, divergence = 0, speed = 0, smoke = 0, heat = 0, acceleration = 0;
        for (int local = threadIdx.x; local < 512; local += blockDim.x) {
            int i = b * 512 + local;
            if (!grid.fluid[i]) continue;
            float div = -grid.solid_divergence[i], velocity_squared = 0;
            if constexpr (full) {
                double h = spacing(grid.layout, i), volume = h * h * h;
                ++fluid;
                smoke += grid.state.smoke[i] * volume;
                heat += grid.state.temperature[i] * volume;
            }
            for (int a = 0; a < 3; ++a) {
                if constexpr (full) div += (side_velocity(grid, grid.state, i, a, 1, true) - side_velocity(grid, grid.state, i, a, -1, true)) / spacing(grid.layout, i);
                float v = fmaxf(fabsf(side_velocity(grid, grid.state, i, a, -1)), fabsf(side_velocity(grid, grid.state, i, a, 1)));
                velocity_squared += v * v;
            }
            if constexpr (full) divergence = fmax(divergence, double(fabsf(div)));
            speed        = fmax(speed, double(sqrtf(velocity_squared)));
            acceleration = fmax(acceleration, double(fabsf(config.buoyancy * grid.state.temperature[i] - config.smoke_weight * grid.state.smoke[i])));
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
    __global__ void finish_measure(Grid grid, const Source* sources, int source_count, float time, double* results) {
        constexpr int count = full ? 6 : 2;
        int k               = threadIdx.x;
        if (k >= count) return;
        double value = 0;
        for (int b = 0; b < grid.layout.count; ++b) value = full && (k == 0 || k == 3 || k == 4) ? value + results[b * count + k] : fmax(value, results[b * count + k]);
        if (k == (full ? 2 : 0)) {
            for (int s = 0; s < source_count; ++s)
                if (time >= sources[s].start_time && time <= sources[s].end_time) value = fmax(value, double(sqrtf(sources[s].velocity.x * sources[s].velocity.x + sources[s].velocity.y * sources[s].velocity.y + sources[s].velocity.z * sources[s].velocity.z) + sources[s].motion.max_speed * sources[s].inherit_velocity));
            value = fmax(value, double(grid.collider_speed));
        }
        results[grid.layout.count * count + k] = value;
    }
    void measure(Grid grid, Config config, const Source* sources, int source_count, float time, bool full, double* reductions, cudaStream_t stream) {
        if (full) {
            if (grid.layout.count) measure_blocks<true><<<grid.layout.count, 256, 0, stream>>>(grid, config, reductions);
            finish_measure<true><<<1, 32, 0, stream>>>(grid, sources, source_count, time, reductions);
        } else {
            if (grid.layout.count) measure_blocks<false><<<grid.layout.count, 256, 0, stream>>>(grid, config, reductions);
            finish_measure<false><<<1, 32, 0, stream>>>(grid, sources, source_count, time, reductions);
        }
    }
} // namespace pyro::cuda::sparse
