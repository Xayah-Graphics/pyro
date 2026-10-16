#ifndef GEOMETRY_QUERY_CUH
#define GEOMETRY_QUERY_CUH

#include <cmath>

#include "geometry.cuh"

namespace geometry {
    __host__ __device__ inline float3 transform_point(const float* matrix, float3 point) {
        return {matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3], matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7], matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11]};
    }

    __host__ __device__ inline Transform transform_at(const AffineMotion& motion, float time) {
        Transform result;
        for (int i = 0; i < 12; ++i) result.matrix[i] = motion.matrix[i] + (time - motion.time) * motion.derivative[i];
        const float* m = result.matrix;
        float3 a{m[0], m[4], m[8]}, b{m[1], m[5], m[9]}, c{m[2], m[6], m[10]};
        float3 u{b.y * c.z - b.z * c.y, b.z * c.x - b.x * c.z, b.x * c.y - b.y * c.x};
        float3 v{c.y * a.z - c.z * a.y, c.z * a.x - c.x * a.z, c.x * a.y - c.y * a.x};
        float3 w{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        float inverse_det     = 1.0f / (a.x * u.x + a.y * u.y + a.z * u.z);
        u                     = {u.x * inverse_det, u.y * inverse_det, u.z * inverse_det};
        v                     = {v.x * inverse_det, v.y * inverse_det, v.z * inverse_det};
        w                     = {w.x * inverse_det, w.y * inverse_det, w.z * inverse_det};
        result.inverse[0]     = u;
        result.inverse[1]     = v;
        result.inverse[2]     = w;
        result.inverse_metric = 1.0f / ((u.x * u.x + u.y * u.y + u.z * u.z) + (v.x * v.x + v.y * v.y + v.z * v.z) + (w.x * w.x + w.y * w.y + w.z * w.z));
        return result;
    }

    __host__ __device__ inline float3 inverse_transform_point(const Transform& pose, float3 point) {
        float3 q{point.x - pose.matrix[3], point.y - pose.matrix[7], point.z - pose.matrix[11]};
        float3 u = pose.inverse[0], v = pose.inverse[1], w = pose.inverse[2];
        return {u.x * q.x + u.y * q.y + u.z * q.z, v.x * q.x + v.y * q.y + v.z * q.z, w.x * q.x + w.y * q.y + w.z * q.z};
    }

    __host__ __device__ inline float4 rotation_at(float4 orientation, float3 angular_velocity, float pose_time, float time) {
        float speed  = sqrtf(angular_velocity.x * angular_velocity.x + angular_velocity.y * angular_velocity.y + angular_velocity.z * angular_velocity.z);
        float angle  = 0.5f * speed * (time - pose_time);
        float factor = speed > 0.0f ? sinf(angle) / speed : 0.0f;
        float3 a{angular_velocity.x * factor, angular_velocity.y * factor, angular_velocity.z * factor};
        float4 q = orientation;
        float3 b{q.x, q.y, q.z};
        float cosine = cosf(angle);
        float3 v{b.x * cosine + a.x * q.w + (a.y * b.z - a.z * b.y), b.y * cosine + a.y * q.w + (a.z * b.x - a.x * b.z), b.z * cosine + a.z * q.w + (a.x * b.y - a.y * b.x)};
        return {v.x, v.y, v.z, cosine * q.w - (a.x * b.x + a.y * b.y + a.z * b.z)};
    }

    __host__ __device__ inline float3 rotate(float4 q, float3 p) {
        float3 inner{q.y * p.z - q.z * p.y + p.x * q.w, q.z * p.x - q.x * p.z + p.y * q.w, q.x * p.y - q.y * p.x + p.z * q.w};
        return {p.x + (q.y * inner.z - q.z * inner.y) * 2.0f, p.y + (q.z * inner.x - q.x * inner.z) * 2.0f, p.z + (q.x * inner.y - q.y * inner.x) * 2.0f};
    }

    template <MotionKind motion_kind, bool derivatives>
    __host__ __device__ inline InstanceState evaluate(const InstanceView& instance, float time) {
        InstanceState result;
        if constexpr (motion_kind == MotionKind::affine) {
            result.transform = transform_at(instance.motion.affine, time);
            if constexpr (derivatives)
                for (int i = 0; i < 12; ++i) result.derivative[i] = instance.motion.affine.derivative[i];
        } else {
            const auto& motion = instance.motion.rigid;
            float3 omega{motion.angular_velocity[0], motion.angular_velocity[1], motion.angular_velocity[2]};
            float4 orientation{motion.orientation[0], motion.orientation[1], motion.orientation[2], motion.orientation[3]};
            auto q = rotation_at(orientation, omega, motion.time, time);
            AffineMotion pose;
            for (int axis = 0; axis < 3; ++axis) {
                float3 column         = rotate(q, {axis == 0 ? motion.scale : 0, axis == 1 ? motion.scale : 0, axis == 2 ? motion.scale : 0});
                pose.matrix[axis]     = column.x;
                pose.matrix[axis + 4] = column.y;
                pose.matrix[axis + 8] = column.z;
                if constexpr (derivatives) {
                    result.derivative[axis]     = omega.y * column.z - omega.z * column.y;
                    result.derivative[axis + 4] = omega.z * column.x - omega.x * column.z;
                    result.derivative[axis + 8] = omega.x * column.y - omega.y * column.x;
                }
                pose.matrix[axis * 4 + 3] = motion.translation[axis] + motion.linear_velocity[axis] * (time - motion.time);
                if constexpr (derivatives) result.derivative[axis * 4 + 3] = motion.linear_velocity[axis];
            }
            result.transform      = transform_at(pose, 0);
            result.distance_scale = motion.scale;
        }
        return result;
    }

    template <bool derivatives = true>
    __host__ __device__ inline InstanceState evaluate(const InstanceView& instance, float time) {
        if (instance.motion.kind == MotionKind::affine) return evaluate<MotionKind::affine, derivatives>(instance, time);
        return evaluate<MotionKind::rigid, derivatives>(instance, time);
    }

    __host__ __device__ inline float3 point_velocity(const InstanceState& state, float3 world) {
        return transform_point(state.derivative, inverse_transform_point(state.transform, world));
    }

    __host__ __device__ inline float velocity_divergence(const InstanceState& state) {
        auto u = state.transform.inverse[0], v = state.transform.inverse[1], w = state.transform.inverse[2];
        const float* d = state.derivative;
        return (d[0] * u.x + d[4] * u.y + d[8] * u.z) + (d[1] * v.x + d[5] * v.y + d[9] * v.z) + (d[2] * w.x + d[6] * w.y + d[10] * w.z);
    }

    __host__ __device__ inline float signed_distance(Sphere sphere, float3 local) {
        return sqrtf(local.x * local.x + local.y * local.y + local.z * local.z) - sphere.radius;
    }

    __host__ __device__ inline float signed_distance(Box box, float3 local) {
        float3 q{fabsf(local.x) - box.half_extent.x, fabsf(local.y) - box.half_extent.y, fabsf(local.z) - box.half_extent.z};
        float3 positive{fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f)};
        return sqrtf(positive.x * positive.x + positive.y * positive.y + positive.z * positive.z) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
    }

    __host__ __device__ inline float signed_distance(Cylinder cylinder, float3 local) {
        float radial         = sqrtf(local.x * local.x + local.y * local.y) - cylinder.radius;
        float axial          = fabsf(local.z) - cylinder.half_height;
        float outside_radial = fmaxf(radial, 0.0f), outside_axial = fmaxf(axial, 0.0f);
        return sqrtf(outside_radial * outside_radial + outside_axial * outside_axial) + fminf(fmaxf(radial, axial), 0.0f);
    }

    __host__ __device__ inline float signed_distance(Cone cone, float3 local) {
        float radial            = sqrtf(local.x * local.x + local.y * local.y);
        float height            = 2.0f * cone.half_height;
        float radius_difference = cone.upper_radius - cone.lower_radius;
        float axial             = local.z + cone.half_height;
        float parameter         = fminf(1.0f, fmaxf(0.0f, ((radial - cone.lower_radius) * radius_difference + axial * height) / (radius_difference * radius_difference + height * height)));
        float side_radial       = radial - (cone.lower_radius + parameter * radius_difference);
        float side_axial        = axial - parameter * height;
        float bottom_radial     = fmaxf(radial - cone.lower_radius, 0.0f);
        float top_radial        = fmaxf(radial - cone.upper_radius, 0.0f);
        float top_axial         = local.z - cone.half_height;
        float squared_distance  = fminf(side_radial * side_radial + side_axial * side_axial, fminf(bottom_radial * bottom_radial + axial * axial, top_radial * top_radial + top_axial * top_axial));
        bool inside             = axial >= 0.0f && axial <= height && radial <= cone.lower_radius + (axial / height) * radius_difference;
        float distance          = sqrtf(squared_distance);
        return inside ? -distance : distance;
    }

    __host__ __device__ inline float signed_distance(Torus torus, float3 local) {
        float radial = sqrtf(local.x * local.x + local.y * local.y) - torus.major_radius;
        return sqrtf(radial * radial + local.z * local.z) - torus.minor_radius;
    }

    __device__ inline int next_node(int node) {
        while (node & 1) node >>= 1;
        return node ? node + 1 : 0;
    }

    __device__ inline float triangle_distance_squared(float3 p, float3 a, float3 b, float3 c) {
        float3 ab{b.x - a.x, b.y - a.y, b.z - a.z}, ac{c.x - a.x, c.y - a.y, c.z - a.z}, ap{p.x - a.x, p.y - a.y, p.z - a.z};
        float d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z, d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
        if (d1 <= 0 && d2 <= 0) return ap.x * ap.x + ap.y * ap.y + ap.z * ap.z;
        float3 bp{p.x - b.x, p.y - b.y, p.z - b.z};
        float d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z, d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
        if (d3 >= 0 && d4 <= d3) return bp.x * bp.x + bp.y * bp.y + bp.z * bp.z;
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            float weight = d1 / (d1 - d3);
            float3 v{ap.x - ab.x * weight, ap.y - ab.y * weight, ap.z - ab.z * weight};
            return v.x * v.x + v.y * v.y + v.z * v.z;
        }
        float3 cp{p.x - c.x, p.y - c.y, p.z - c.z};
        float d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z, d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
        if (d6 >= 0 && d5 <= d6) return cp.x * cp.x + cp.y * cp.y + cp.z * cp.z;
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            float weight = d2 / (d2 - d6);
            float3 v{ap.x - ac.x * weight, ap.y - ac.y * weight, ap.z - ac.z * weight};
            return v.x * v.x + v.y * v.y + v.z * v.z;
        }
        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
            float weight = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            float3 v{bp.x - (c.x - b.x) * weight, bp.y - (c.y - b.y) * weight, bp.z - (c.z - b.z) * weight};
            return v.x * v.x + v.y * v.y + v.z * v.z;
        }
        float weight_b = vb / (va + vb + vc), weight_c = vc / (va + vb + vc);
        float3 v{ap.x - ab.x * weight_b - ac.x * weight_c, ap.y - ab.y * weight_b - ac.y * weight_c, ap.z - ab.z * weight_b - ac.z * weight_c};
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    __device__ inline float mesh_signed_distance(MeshView mesh, const Transform& transform, float3 world, float max_distance) {
        BvhView bvh        = mesh.bvh;
        const float* m     = transform.matrix;
        float3 p           = inverse_transform_point(transform, world);
        float metric       = transform.inverse_metric;
        float best         = max_distance * max_distance;
        bool inside_bounds = p.x >= mesh.bounds.lower.x && p.y >= mesh.bounds.lower.y && p.z >= mesh.bounds.lower.z && p.x <= mesh.bounds.upper.x && p.y <= mesh.bounds.upper.y && p.z <= mesh.bounds.upper.z;
        for (int node = 1; node;) {
            Bounds bounds = bvh.nodes[node];
            float3 delta{fmaxf(bounds.lower.x - p.x, fmaxf(p.x - bounds.upper.x, 0.0f)), fmaxf(bounds.lower.y - p.y, fmaxf(p.y - bounds.upper.y, 0.0f)), fmaxf(bounds.lower.z - p.z, fmaxf(p.z - bounds.upper.z, 0.0f))};
            if ((delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) * metric >= best) {
                node = next_node(node);
                continue;
            }
            if (node < bvh.leaf_count) {
                node *= 2;
                continue;
            }
            int leaf = node - bvh.leaf_count;
            if (leaf < mesh.triangle_count) {
                int3 triangle = mesh.triangles[bvh.order[leaf]];
                float3 ta = mesh.positions[triangle.x], tb = mesh.positions[triangle.y], tc = mesh.positions[triangle.z];
                best = fminf(best, triangle_distance_squared(world, transform_point(m, ta), transform_point(m, tb), transform_point(m, tc)));
            }
            node = next_node(node);
        }
        int crossings = 0;
        if (inside_bounds) {
            constexpr double dy = 0.3713906763541037, dz = 0.1278312937608921;
            for (int node = 1; node;) {
                Bounds bounds = bvh.nodes[node];
                double entry  = fmax(0.0, fmax(static_cast<double>(bounds.lower.x - p.x), fmax((bounds.lower.y - p.y) / dy, (bounds.lower.z - p.z) / dz)));
                double exit   = fmin(static_cast<double>(bounds.upper.x - p.x), fmin((bounds.upper.y - p.y) / dy, (bounds.upper.z - p.z) / dz));
                if (entry > exit) {
                    node = next_node(node);
                    continue;
                }
                if (node < bvh.leaf_count) {
                    node *= 2;
                    continue;
                }
                int leaf = node - bvh.leaf_count;
                if (leaf < mesh.triangle_count) {
                    int3 triangle = mesh.triangles[bvh.order[leaf]];
                    float3 ta = mesh.positions[triangle.x], tb = mesh.positions[triangle.y], tc = mesh.positions[triangle.z];
                    double ex = double(tb.x) - ta.x, ey = double(tb.y) - ta.y, ez = double(tb.z) - ta.z;
                    double fx = double(tc.x) - ta.x, fy = double(tc.y) - ta.y, fz = double(tc.z) - ta.z;
                    double hx = dy * fz - dz * fy, hy = dz * fx - fz, hz = fy - dy * fx;
                    double denominator = ex * hx + ey * hy + ez * hz;
                    if (denominator != 0) {
                        double sx = double(p.x) - ta.x, sy = double(p.y) - ta.y, sz = double(p.z) - ta.z;
                        double r  = (sx * hx + sy * hy + sz * hz) / denominator;
                        double qx = sy * ez - sz * ey, qy = sz * ex - sx * ez, qz = sx * ey - sy * ex;
                        double s        = (qx + dy * qy + dz * qz) / denominator;
                        double distance = (fx * qx + fy * qy + fz * qz) / denominator;
                        if (r >= 0 && s >= 0 && r + s <= 1 && distance > 0) ++crossings;
                    }
                }
                node = next_node(node);
            }
        }
        return (crossings & 1 ? -sqrtf(best) : sqrtf(best));
    }

    template <ShapeKind shape_kind>
    __device__ inline float signed_distance(const InstanceView& instance, const InstanceState& state, float3 world, float max_distance) {
        auto bounds = instance.swept;
        if (world.x < bounds.lower.x - max_distance || world.y < bounds.lower.y - max_distance || world.z < bounds.lower.z - max_distance || world.x > bounds.upper.x + max_distance || world.y > bounds.upper.y + max_distance || world.z > bounds.upper.z + max_distance) return max_distance;
        if constexpr (shape_kind == ShapeKind::mesh) return mesh_signed_distance(instance.shape.mesh, state.transform, world, max_distance);
        else {
            float3 local = inverse_transform_point(state.transform, world);
            float distance;
            if constexpr (shape_kind == ShapeKind::sphere) distance = signed_distance(instance.shape.sphere, local);
            else if constexpr (shape_kind == ShapeKind::box) distance = signed_distance(instance.shape.box, local);
            else if constexpr (shape_kind == ShapeKind::cylinder) distance = signed_distance(instance.shape.cylinder, local);
            else if constexpr (shape_kind == ShapeKind::cone) distance = signed_distance(instance.shape.cone, local);
            else if constexpr (shape_kind == ShapeKind::torus) distance = signed_distance(instance.shape.torus, local);
            return fminf(max_distance, fmaxf(-max_distance, distance * state.distance_scale));
        }
    }

    __device__ inline float signed_distance(const InstanceView& instance, const InstanceState& state, float3 world, float max_distance) {
        float distance;
        switch (instance.shape.kind) {
        case ShapeKind::sphere: distance = signed_distance<ShapeKind::sphere>(instance, state, world, max_distance); break;
        case ShapeKind::box: distance = signed_distance<ShapeKind::box>(instance, state, world, max_distance); break;
        case ShapeKind::cylinder: distance = signed_distance<ShapeKind::cylinder>(instance, state, world, max_distance); break;
        case ShapeKind::cone: distance = signed_distance<ShapeKind::cone>(instance, state, world, max_distance); break;
        case ShapeKind::torus: distance = signed_distance<ShapeKind::torus>(instance, state, world, max_distance); break;
        case ShapeKind::mesh: distance = signed_distance<ShapeKind::mesh>(instance, state, world, max_distance); break;
        }
        return distance;
    }
}
#endif
