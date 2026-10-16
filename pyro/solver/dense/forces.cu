#include "forces.cuh"

#include <algorithm>
#include <cmath>

#include <cuda/launch>

#include <solver/common/scene.cuh>

namespace pyro::dense {
    template <bool buoyant>
    __global__ void forces_kernel(Grid grid, ForcesConfig config, float3 wind, Fields destination, Fields force_fields, float dt, float smoke_decay, float heat_decay) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i                      = blockIdx.x * tile_cells + local;
            int3 c                     = cell_coordinate(grid.layout, i);
            destination.smoke[i]       = grid.fluid[i] ? destination.smoke[i] * smoke_decay : 0;
            destination.temperature[i] = grid.fluid[i] ? destination.temperature[i] * heat_decay : 0;
            for (int axis = 0; axis < 3; ++axis) {
                if (!face_valid(grid.layout, c, axis)) {
                    destination.velocity[axis][i] = (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z);
                    continue;
                }
                if constexpr (buoyant) {
                    int previous      = neighbor(grid.layout, i, axis, -1);
                    float density     = 0.5f * (force_fields.smoke[i] + value(force_fields.smoke, previous));
                    float temperature = 0.5f * (force_fields.temperature[i] + value(force_fields.temperature, previous));
                    destination.velocity[axis][i] += (axis == 0 ? config.up.x : axis == 1 ? config.up.y : config.up.z) * (config.buoyancy * temperature - config.smoke_weight * density) * dt;
                }
            }
        }
    }

    template <::geometry::ShapeKind shape_kind, ::geometry::MotionKind motion_kind>
    __global__ void cache_source_kernel(Grid grid, float cell_size, const __grid_constant__ SourceView source, SourceRegion region) {
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        int3 extent = region.extent;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        int3 c{region.lower.x + ordinal % extent.x, region.lower.y + ordinal / extent.x % extent.y, region.lower.z + ordinal / (extent.x * extent.y)};
        float3 p{grid.origin.x + (c.x + .5f) * cell_size, grid.origin.y + (c.y + .5f) * cell_size, grid.origin.z + (c.z + .5f) * cell_size};
        region.weights[ordinal * 4] = source_weight<shape_kind, motion_kind>(source, p, cell_size, 0.0f);
        for (int axis = 0; axis < 3; ++axis) {
            float3 face = p;
            if (axis == 0) face.x -= cell_size * .5f;
            if (axis == 1) face.y -= cell_size * .5f;
            if (axis == 2) face.z -= cell_size * .5f;
            region.weights[ordinal * 4 + axis + 1] = source_weight<shape_kind, motion_kind>(source, face, cell_size, 0.0f);
        }
    }

    template <bool cached, bool scalars, bool velocity, bool inherited, ::geometry::ShapeKind shape_kind = ::geometry::ShapeKind::sphere, ::geometry::MotionKind motion_kind = ::geometry::MotionKind::rigid>
    __global__ void source_kernel(Grid grid, float cell_size, Fields destination, const __grid_constant__ SourceView source, SourceRegion region, float sample_time, float overlap) {
        int3 lower = region.lower, extent = region.extent;
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        if (ordinal >= extent.x * extent.y * extent.z) return;
        int3 c{lower.x + ordinal % extent.x, lower.y + ordinal / extent.x % extent.y, lower.z + ordinal / (extent.x * extent.y)};
        int i = cell_index(grid.layout, c);
        float3 p{grid.origin.x + (c.x + 0.5f) * cell_size, grid.origin.y + (c.y + 0.5f) * cell_size, grid.origin.z + (c.z + 0.5f) * cell_size};
        if constexpr (scalars)
            if (grid.fluid[i]) {
                float weight = (cached ? region.weights[ordinal * 4] : source_weight<shape_kind, motion_kind>(source, p, cell_size, sample_time)) * overlap;
                destination.smoke[i] += source.smoke_rate * weight;
                destination.temperature[i] += source.temperature_rate * weight;
            }
        if constexpr (velocity)
            for (int axis = 0; axis < 3; ++axis) {
                if (!face_valid(grid.layout, c, axis)) continue;
                float3 face = p;
                if (axis == 0) face.x -= cell_size * 0.5f;
                if (axis == 1) face.y -= cell_size * 0.5f;
                if (axis == 2) face.z -= cell_size * 0.5f;
                float3 motion_velocity{};
                float weight                  = cached ? region.weights[ordinal * 4 + axis + 1] : source_weight<shape_kind, motion_kind, inherited>(source, face, cell_size, sample_time, &motion_velocity);
                float blend                   = -expm1f(-source.velocity_rate * weight * overlap);
                float current_velocity        = destination.velocity[axis][i];
                float target                  = axis == 0 ? source.velocity.x + motion_velocity.x * source.inherit_velocity : axis == 1 ? source.velocity.y + motion_velocity.y * source.inherit_velocity : source.velocity.z + motion_velocity.z * source.inherit_velocity;
                destination.velocity[axis][i] = current_velocity + blend * (target - current_velocity);
            }
    }

    __global__ void decay_kernel(Fields destination, float smoke_decay, float heat_decay) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            destination.smoke[i] *= smoke_decay;
            destination.temperature[i] *= heat_decay;
        }
    }

    void cache_source(Grid grid, float cell_size, const SourceView& source, SourceRegion region, ::cuda::stream_ref stream) {
        int cells = region.extent.x * region.extent.y * region.extent.z;
        if (!cells) return;
        auto launch = [&]<::geometry::ShapeKind shape_kind, ::geometry::MotionKind motion_kind = ::geometry::MotionKind::rigid>() { ::cuda::launch(stream, ::cuda::distribute<threads>(cells), cache_source_kernel<shape_kind, motion_kind>, grid, cell_size, source, region); };
        switch (source.instance.shape.kind) {
        case ::geometry::ShapeKind::sphere: launch.template operator()<::geometry::ShapeKind::sphere>(); break;
        case ::geometry::ShapeKind::box: launch.template operator()<::geometry::ShapeKind::box>(); break;
        case ::geometry::ShapeKind::cylinder: launch.template operator()<::geometry::ShapeKind::cylinder>(); break;
        case ::geometry::ShapeKind::cone: launch.template operator()<::geometry::ShapeKind::cone>(); break;
        case ::geometry::ShapeKind::torus: launch.template operator()<::geometry::ShapeKind::torus>(); break;
        case ::geometry::ShapeKind::mesh:
            if (source.instance.motion.kind == ::geometry::MotionKind::affine) launch.template operator()<::geometry::ShapeKind::mesh, ::geometry::MotionKind::affine>();
            else launch.template operator()<::geometry::ShapeKind::mesh>();
            break;
        }
    }

    void apply_sources_forces(Grid grid, ForcesConfig config, float cell_size, float3 wind, Fields destination, Fields force_fields, const SourceRegion* regions, std::span<const Source> inputs, std::span<const ::geometry::InstanceView> instances, float time, float dt, ::cuda::stream_ref stream) {
        float smoke_decay = std::exp(-.5f * config.dissipation * dt), heat_decay = std::exp(-.5f * config.cooling * dt);
        if ((config.buoyancy != 0 || config.smoke_weight != 0) && (config.up.x != 0 || config.up.y != 0 || config.up.z != 0)) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), forces_kernel<true>, grid, config, wind, destination, force_fields, dt, smoke_decay, heat_decay);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), forces_kernel<false>, grid, config, wind, destination, force_fields, dt, smoke_decay, heat_decay);
        for (int source = 0; source < static_cast<int>(inputs.size()); ++source) {
            const auto& input = inputs[source];
            float overlap     = std::max(0.f, std::min(time + dt, input.end_time) - std::max(time, input.start_time));
            if (overlap == 0) continue;
            auto region = regions[source];
            int cells   = region.extent.x * region.extent.y * region.extent.z;
            if (!cells || (input.smoke_rate == 0 && input.temperature_rate == 0 && input.velocity_rate == 0)) continue;
            SourceView source_view{instances[source], input.velocity, input.smoke_rate, input.temperature_rate, input.velocity_rate, input.start_time, input.end_time, input.inherit_velocity};
            auto launch = [&]<bool cached, bool inherited, ::geometry::ShapeKind shape_kind = ::geometry::ShapeKind::sphere, ::geometry::MotionKind motion_kind = ::geometry::MotionKind::rigid>() {
                if (input.velocity_rate == 0) ::cuda::launch(stream, ::cuda::distribute<threads>(cells), source_kernel<cached, true, false, false, shape_kind, motion_kind>, grid, cell_size, destination, source_view, region, time + dt * .5f, overlap);
                else if (input.smoke_rate == 0 && input.temperature_rate == 0) ::cuda::launch(stream, ::cuda::distribute<threads>(cells), source_kernel<cached, false, true, inherited, shape_kind, motion_kind>, grid, cell_size, destination, source_view, region, time + dt * .5f, overlap);
                else ::cuda::launch(stream, ::cuda::distribute<threads>(cells), source_kernel<cached, true, true, inherited, shape_kind, motion_kind>, grid, cell_size, destination, source_view, region, time + dt * .5f, overlap);
            };
            if (region.cached) launch.template operator()<true, false>();
            else {
                auto dispatch = [&]<::geometry::ShapeKind shape_kind, ::geometry::MotionKind motion_kind = ::geometry::MotionKind::rigid>() {
                    if (input.inherit_velocity == 0) launch.template operator()<false, false, shape_kind, motion_kind>();
                    else launch.template operator()<false, true, shape_kind, motion_kind>();
                };
                switch (source_view.instance.shape.kind) {
                case ::geometry::ShapeKind::sphere: dispatch.template operator()<::geometry::ShapeKind::sphere>(); break;
                case ::geometry::ShapeKind::box: dispatch.template operator()<::geometry::ShapeKind::box>(); break;
                case ::geometry::ShapeKind::cylinder: dispatch.template operator()<::geometry::ShapeKind::cylinder>(); break;
                case ::geometry::ShapeKind::cone: dispatch.template operator()<::geometry::ShapeKind::cone>(); break;
                case ::geometry::ShapeKind::torus: dispatch.template operator()<::geometry::ShapeKind::torus>(); break;
                case ::geometry::ShapeKind::mesh:
                    if (source_view.instance.motion.kind == ::geometry::MotionKind::affine) dispatch.template operator()<::geometry::ShapeKind::mesh, ::geometry::MotionKind::affine>();
                    else dispatch.template operator()<::geometry::ShapeKind::mesh>();
                    break;
                }
            }
        }
        if (smoke_decay != 1 || heat_decay != 1) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), decay_kernel, destination, smoke_decay, heat_decay);
    }

}
