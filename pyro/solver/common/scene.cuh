#ifndef PYRO_SOLVER_COMMON_SCENE_CUH
#define PYRO_SOLVER_COMMON_SCENE_CUH

#include <geometry/query.cuh>

namespace pyro {
    enum struct BoundaryMode { none, stationary, moving };

    struct Source {
        int instance{};
        float3 velocity{};
        float smoke_rate{1.0f};
        float temperature_rate{1.0f};
        float velocity_rate{4.0f};
        float start_time{};
        float end_time{1.0e20f};
        float inherit_velocity{};

        bool operator==(const Source& other) const {
            return instance == other.instance && velocity.x == other.velocity.x && velocity.y == other.velocity.y && velocity.z == other.velocity.z && smoke_rate == other.smoke_rate && temperature_rate == other.temperature_rate && velocity_rate == other.velocity_rate && start_time == other.start_time && end_time == other.end_time && inherit_velocity == other.inherit_velocity;
        }
    };

    struct Collider {
        int instance{};
        float surface_offset{};

        bool operator==(const Collider&) const = default;
    };

    struct SourceView {
        ::geometry::InstanceView instance;
        float3 velocity{};
        float smoke_rate{1.0f};
        float temperature_rate{1.0f};
        float velocity_rate{4.0f};
        float start_time{};
        float end_time{1.0e20f};
        float inherit_velocity{};
    };

    struct ColliderView {
        ::geometry::InstanceView instance;
        float surface_offset{};
    };

    template <bool velocity_output = false, bool divergence_output = false>
    __device__ inline float solid_distance(float3 world, float time, const ColliderView* bodies, int count, float max_distance, float3* velocity = nullptr, float* divergence = nullptr) {
        float distance = 1.0e20f;
        float3 speed{};
        float rate = 0;
        for (int i = 0; i < count; ++i) {
            const auto& body = bodies[i];
            auto state       = ::geometry::evaluate<(velocity_output || divergence_output)>(body.instance, time);
            float band       = max_distance + fabsf(body.surface_offset);
            float d          = ::geometry::signed_distance(body.instance, state, world, band) - body.surface_offset;
            if (d < distance) {
                distance = d;
                if constexpr (velocity_output) speed = ::geometry::point_velocity(state, world);
                if constexpr (divergence_output) rate = ::geometry::velocity_divergence(state);
            }
        }
        if constexpr (velocity_output) *velocity = speed;
        if constexpr (divergence_output) *divergence = rate;
        return distance;
    }

    template <bool velocity_output = false>
    __device__ inline float source_weight(const SourceView& source, float3 p, float h, float time, float3* motion_velocity = nullptr) {
        auto state     = ::geometry::evaluate<velocity_output>(source.instance, time);
        float distance = ::geometry::signed_distance(source.instance, state, p, .5f * h);
        if constexpr (velocity_output) *motion_velocity = ::geometry::point_velocity(state, p);
        return fminf(1.0f, fmaxf(0.5f - distance / h, 0.0f));
    }

    template <::geometry::ShapeKind shape_kind, ::geometry::MotionKind motion_kind, bool velocity_output = false>
    __device__ inline float source_weight(const SourceView& source, float3 p, float h, float time, float3* motion_velocity = nullptr) {
        auto state     = ::geometry::evaluate<motion_kind, velocity_output>(source.instance, time);
        float distance = ::geometry::signed_distance<shape_kind>(source.instance, state, p, .5f * h);
        if constexpr (velocity_output) *motion_velocity = ::geometry::point_velocity(state, p);
        return fminf(1.f, fmaxf(.5f - distance / h, 0.f));
    }
}
#endif
