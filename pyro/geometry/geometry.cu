#include "geometry.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <cuda/algorithm>
#include <cuda/std/span>

#include "bvh.cuh"
#include "query.cuh"

namespace geometry {
    Mesh::Mesh(std::span<const float3> points, std::span<const int3> faces, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool)
        : positions(::cuda::make_buffer<float3>(stream, pool, points.size(), ::cuda::no_init)),
          triangles(::cuda::make_buffer<int3>(stream, pool, faces.size(), ::cuda::no_init)),
          order(::cuda::make_buffer<int>(stream, pool, faces.size(), ::cuda::no_init)),
          nodes(::cuda::make_buffer<Bounds>(stream, pool, std::bit_ceil(faces.size()) * 2, ::cuda::no_init)) {
        data = {positions.data(), triangles.data(), static_cast<int>(faces.size()), {{1.e30f, 1.e30f, 1.e30f}, {-1.e30f, -1.e30f, -1.e30f}}, {nodes.data(), order.data(), static_cast<int>(nodes.size() / 2)}};
        for (const auto& point : points) {
            data.bounds.lower = {std::min(data.bounds.lower.x, point.x), std::min(data.bounds.lower.y, point.y), std::min(data.bounds.lower.z, point.z)};
            data.bounds.upper = {std::max(data.bounds.upper.x, point.x), std::max(data.bounds.upper.y, point.y), std::max(data.bounds.upper.z, point.z)};
        }
        ::cuda::copy_bytes(stream, ::cuda::std::span(points.data(), points.size()), positions, {.src_access_order = ::cuda::source_access_order::during_api_call});
        ::cuda::copy_bytes(stream, ::cuda::std::span(faces.data(), faces.size()), triangles, {.src_access_order = ::cuda::source_access_order::during_api_call});
        build_bvh(data, stream, pool);
    }

    Scene::Scene(::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool)
        : stream(stream),
          pool(pool) {}

    int Scene::add_mesh(std::span<const float3> positions, std::span<const int3> triangles) {
        int index        = static_cast<int>(meshes.size());
        const auto& mesh = meshes.emplace_back(positions, triangles, stream, pool);
        shapes.push_back({.kind = ShapeKind::mesh, .mesh = index});
        return static_cast<int>(shapes.size()) - 1;
    }

    void Scene::prepare(float start_time, float end_time) {
        prepared.resize(instances.size());
        for (std::size_t index = 0; index < instances.size(); ++index) {
            const auto& instance = instances[index];
            const auto& shape    = shapes[instance.shape];
            auto& result         = prepared[index];
            result               = {.motion = instance.motion};
            if (result.motion.kind == MotionKind::rigid) {
                const auto& motion = result.motion.rigid;
                bool moving        = false;
                for (int axis = 0; axis < 3; ++axis) moving |= motion.linear_velocity[axis] != 0 || motion.angular_velocity[axis] != 0;
                if (!moving) result.motion.rigid.time = 0;
            } else if (std::ranges::none_of(result.motion.affine.derivative, [](float value) { return value != 0; })) result.motion.affine.time = 0;
            switch (shape.kind) {
            case ShapeKind::sphere: result.shape = {.kind = ShapeKind::sphere, .sphere = shape.sphere}; break;
            case ShapeKind::box: result.shape = {.kind = ShapeKind::box, .box = shape.box}; break;
            case ShapeKind::cylinder: result.shape = {.kind = ShapeKind::cylinder, .cylinder = shape.cylinder}; break;
            case ShapeKind::cone: result.shape = {.kind = ShapeKind::cone, .cone = shape.cone}; break;
            case ShapeKind::torus: result.shape = {.kind = ShapeKind::torus, .torus = shape.torus}; break;
            case ShapeKind::mesh: result.shape = {.kind = ShapeKind::mesh, .mesh = meshes[shape.mesh].data}; break;
            }
            if (result.motion.kind == MotionKind::affine && result.shape.kind != ShapeKind::mesh) throw std::runtime_error("Analytic primitives require rigid motion with uniform scale");
            Bounds local;
            switch (result.shape.kind) {
            case ShapeKind::sphere:
                {
                    float r = result.shape.sphere.radius;
                    local   = {{-r, -r, -r}, {r, r, r}};
                    break;
                }
            case ShapeKind::box:
                {
                    const auto& h = result.shape.box.half_extent;
                    local         = {{-h.x, -h.y, -h.z}, {h.x, h.y, h.z}};
                    break;
                }
            case ShapeKind::cylinder:
                {
                    const auto& cylinder = result.shape.cylinder;
                    local                = {{-cylinder.radius, -cylinder.radius, -cylinder.half_height}, {cylinder.radius, cylinder.radius, cylinder.half_height}};
                    break;
                }
            case ShapeKind::cone:
                {
                    const auto& cone = result.shape.cone;
                    float radius     = std::max(cone.lower_radius, cone.upper_radius);
                    local            = {{-radius, -radius, -cone.half_height}, {radius, radius, cone.half_height}};
                    break;
                }
            case ShapeKind::torus:
                {
                    const auto& torus = result.shape.torus;
                    float radius      = torus.major_radius + torus.minor_radius;
                    local             = {{-radius, -radius, -torus.minor_radius}, {radius, radius, torus.minor_radius}};
                    break;
                }
            case ShapeKind::mesh: local = result.shape.mesh.bounds; break;
            }
            result.swept = {{1.e30f, 1.e30f, 1.e30f}, {-1.e30f, -1.e30f, -1.e30f}};
            auto first = evaluate<true>(result, start_time), last = evaluate<false>(result, end_time);
            float radius = 0, affine_speed = 0;
            for (int corner = 0; corner < 8; ++corner) {
                float3 point{corner & 1 ? local.upper.x : local.lower.x, corner & 2 ? local.upper.y : local.lower.y, corner & 4 ? local.upper.z : local.lower.z};
                radius        = std::max(radius, std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z));
                auto velocity = transform_point(first.derivative, point);
                affine_speed  = std::max(affine_speed, std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z));
                for (const auto& state : {first, last}) {
                    auto p             = transform_point(state.transform.matrix, point);
                    result.swept.lower = {std::min(result.swept.lower.x, p.x), std::min(result.swept.lower.y, p.y), std::min(result.swept.lower.z, p.z)};
                    result.swept.upper = {std::max(result.swept.upper.x, p.x), std::max(result.swept.upper.y, p.y), std::max(result.swept.upper.z, p.z)};
                }
            }
            if (result.motion.kind == MotionKind::rigid) {
                const auto& motion = result.motion.rigid;
                float linear = 0, angular = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    linear += motion.linear_velocity[axis] * motion.linear_velocity[axis];
                    angular += motion.angular_velocity[axis] * motion.angular_velocity[axis];
                }
                result.moving = linear != 0 || angular != 0;
                radius *= motion.scale;
                result.max_speed = std::sqrt(linear) + std::sqrt(angular) * radius;
                if (angular != 0) {
                    float3 a{first.transform.matrix[3], first.transform.matrix[7], first.transform.matrix[11]}, b{last.transform.matrix[3], last.transform.matrix[7], last.transform.matrix[11]};
                    result.swept = {{std::min(a.x, b.x) - radius, std::min(a.y, b.y) - radius, std::min(a.z, b.z) - radius}, {std::max(a.x, b.x) + radius, std::max(a.y, b.y) + radius, std::max(a.z, b.z) + radius}};
                }
            } else {
                result.moving    = std::ranges::any_of(result.motion.affine.derivative, [](float value) { return value != 0; });
                result.max_speed = affine_speed;
            }
            if (result.max_speed > 0) result.max_speed = std::nextafter(result.max_speed, std::numeric_limits<float>::infinity());
        }
    }

}
