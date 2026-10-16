#include "forces.cuh"

#include <cmath>

#include <cuda/launch>

#include <solver/common/scene.cuh>

namespace pyro::sparse {
    template <bool has_sources>
    __global__ void scalar_forces(Grid grid, Fields destination, const SourceView* sources, int count, float time, float dt, float smoke_decay, float heat_decay) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        float smoke = destination.smoke[i] * smoke_decay, temperature = destination.temperature[i] * heat_decay;
        if constexpr (has_sources)
            for (int s = 0; s < count; ++s) {
                SourceView source = sources[s];
                float active      = fmaxf(0, fminf(time + dt, source.end_time) - fmaxf(time, source.start_time));
                if (!active || (source.smoke_rate == 0 && source.temperature_rate == 0)) continue;
                float weight = source_weight(source, center(grid.layout, i), spacing(grid.layout, i), time + dt * .5f);
                smoke += source.smoke_rate * weight * active;
                temperature += source.temperature_rate * weight * active;
            }
        destination.smoke[i]       = grid.fluid[i] ? smoke * smoke_decay : 0;
        destination.temperature[i] = grid.fluid[i] ? temperature * heat_decay : 0;
    }

    template <bool has_sources, bool inherited, bool buoyant>
    __global__ void velocity_forces(Grid grid, Fields destination, Fields carrier, ForcesConfig config, const SourceView* sources, int count, float time, float dt) {
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
                float weight = source_weight<inherited>(source, p, h, time + dt * .5f, &motion);
                float target = axis == 0 ? source.velocity.x + motion.x * source.inherit_velocity : axis == 1 ? source.velocity.y + motion.y * source.inherit_velocity : source.velocity.z + motion.z * source.inherit_velocity;
                v += (target - v) * (1 - expf(-source.velocity_rate * weight * active));
            }
        destination.velocity[f] = v;
    }

    void forces(Grid grid, Fields destination, Fields carrier, ForcesConfig config, const SourceView* sources, std::span<const Source> inputs, float time, float dt, ::cuda::stream_ref stream) {
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
        if (scalar_sources) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), scalar_forces<true>, grid, destination, sources, count, time, dt, smoke_decay, heat_decay);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), scalar_forces<false>, grid, destination, sources, count, time, dt, smoke_decay, heat_decay);
        auto launch = [&]<bool has_sources, bool inherited>() {
            if (buoyant) ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), velocity_forces<has_sources, inherited, true>, grid, destination, carrier, config, sources, count, time, dt);
            else if constexpr (has_sources) ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), velocity_forces<has_sources, inherited, false>, grid, destination, carrier, config, sources, count, time, dt);
        };
        if (velocity_sources) {
            if (inherited) launch.template operator()<true, true>();
            else launch.template operator()<true, false>();
        } else if (buoyant) launch.template operator()<false, false>();
    }

}
