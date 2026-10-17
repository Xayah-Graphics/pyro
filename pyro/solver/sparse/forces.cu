#include "forces.cuh"

#include <cmath>

#include <cuda/launch>

#include <solver/common/scene.cuh>

namespace pyro::sparse {
    __global__ void source_weights(Grid grid, const SourceView* sources, float time, float* weights) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int cells = grid.layout.count * 512, stride = grid.layout.count * 2240;
        if (i >= stride) return;
        float3 p;
        float h;
        if (i < cells) {
            p = center(grid.layout, i);
            h = spacing(grid.layout, i);
        } else {
            int axis, owner, sign;
            face_position(grid.layout, i - cells, p, h, axis, owner, sign);
        }
        weights[static_cast<std::size_t>(blockIdx.y) * stride + i] = source_weight(sources[blockIdx.y], p, h, time);
    }

    template <bool has_sources, bool cached>
    __global__ void scalar_forces(Grid grid, Fields destination, const SourceView* sources, int count, const float* weights, float time, float dt, float smoke_decay, float heat_decay) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        float smoke = destination.smoke[i] * smoke_decay, temperature = destination.temperature[i] * heat_decay;
        if constexpr (has_sources)
            for (int s = 0; s < count; ++s) {
                SourceView source = sources[s];
                float active      = fmaxf(0, fminf(time + dt, source.end_time) - fmaxf(time, source.start_time));
                if (!active || (source.smoke_rate == 0 && source.temperature_rate == 0)) continue;
                float weight;
                if constexpr (cached) weight = weights[static_cast<std::size_t>(s) * grid.layout.count * 2240 + i];
                else weight = source_weight(source, center(grid.layout, i), spacing(grid.layout, i), time + dt * .5f);
                if (weight == 0) continue;
                smoke += source.smoke_rate * weight * active;
                temperature += source.temperature_rate * weight * active;
            }
        destination.smoke[i]       = grid.fluid[i] ? smoke * smoke_decay : 0;
        destination.temperature[i] = grid.fluid[i] ? temperature * heat_decay : 0;
    }

    template <bool has_sources, bool inherited, bool buoyant, bool cached>
    __global__ void velocity_forces(Grid grid, Fields destination, Fields carrier, ForcesConfig config, const SourceView* sources, int count, const float* weights, float time, float dt) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728 || grid.faces[f].area <= 0) return;
        Face face = grid.faces[f];
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        float v = destination.velocity[f];
        if constexpr (buoyant) {
            float smoke       = .5f * ((face.left >= 0 ? carrier.smoke[face.left] : 0) + (face.right >= 0 ? carrier.smoke[face.right] : 0));
            float temperature = .5f * ((face.left >= 0 ? carrier.temperature[face.left] : 0) + (face.right >= 0 ? carrier.temperature[face.right] : 0));
            v += (axis == 0 ? config.up.x : axis == 1 ? config.up.y : config.up.z) * (config.buoyancy * temperature - config.smoke_weight * smoke) * dt;
        }
        if constexpr (has_sources)
            for (int s = 0; s < count; ++s) {
                SourceView source = sources[s];
                float active      = fmaxf(0, fminf(time + dt, source.end_time) - fmaxf(time, source.start_time));
                if (!active || source.velocity_rate == 0) continue;
                float3 motion{};
                float weight;
                if constexpr (cached) weight = weights[static_cast<std::size_t>(s) * grid.layout.count * 2240 + grid.layout.count * 512 + f];
                else weight = source_weight<inherited>(source, p, h, time + dt * .5f, &motion);
                if (weight == 0) continue;
                float target = axis == 0 ? source.velocity.x + motion.x * source.inherit_velocity : axis == 1 ? source.velocity.y + motion.y * source.inherit_velocity : source.velocity.z + motion.z * source.inherit_velocity;
                v += (target - v) * (1 - expf(-source.velocity_rate * weight * active));
            }
        destination.velocity[f] = v;
    }

    void sample_sources(Grid grid, const SourceView* sources, int count, float time, float* weights, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3((grid.layout.count * 2240 + 255) / 256, count)), ::cuda::block_dims<256>()), source_weights, grid, sources, time, weights);
    }

    void forces(Grid grid, Fields destination, Fields carrier, ForcesConfig config, const SourceView* sources, std::span<const Source> inputs, const float* weights, float time, float dt, ::cuda::stream_ref stream) {
        int count         = static_cast<int>(inputs.size());
        float smoke_decay = std::exp(-.5f * config.dissipation * dt), heat_decay = std::exp(-.5f * config.cooling * dt);
        bool scalar_sources = false, velocity_sources = false, inherited = false;
        for (const auto& source : inputs) {
            if (time >= source.end_time || time + dt <= source.start_time) continue;
            scalar_sources |= source.smoke_rate != 0 || source.temperature_rate != 0;
            velocity_sources |= source.velocity_rate != 0;
            inherited |= source.velocity_rate != 0 && source.inherit_velocity != 0;
        }
        bool buoyant = (config.buoyancy != 0 || config.smoke_weight != 0) && (config.up.x != 0 || config.up.y != 0 || config.up.z != 0);
        auto dispatch = [&]<bool cached>() {
            if (scalar_sources) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), scalar_forces<true, cached>, grid, destination, sources, count, weights, time, dt, smoke_decay, heat_decay);
            else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), scalar_forces<false, cached>, grid, destination, sources, count, weights, time, dt, smoke_decay, heat_decay);
            auto launch = [&]<bool has_sources, bool inherited>() {
                if (buoyant) ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), velocity_forces<has_sources, inherited, true, cached>, grid, destination, carrier, config, sources, count, weights, time, dt);
                else if constexpr (has_sources) ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), velocity_forces<has_sources, inherited, false, cached>, grid, destination, carrier, config, sources, count, weights, time, dt);
            };
            if (velocity_sources) {
                if (inherited) launch.template operator()<true, true>();
                else launch.template operator()<true, false>();
            } else if (buoyant) launch.template operator()<false, false>();
        };
        if (weights) dispatch.template operator()<true>();
        else dispatch.template operator()<false>();
    }

}
