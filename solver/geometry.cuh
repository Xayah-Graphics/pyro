#ifndef PYRO_GEOMETRY_CUH
#define PYRO_GEOMETRY_CUH
#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cmath>
#include <stdexcept>
#endif
namespace pyro::cuda {
    enum struct Shape { sphere, box, mesh };

    struct Triangle {
        float3 a{}, b{}, c{};
    };

    struct Bounds {
        float3 lower{}, upper{};
    };

    struct Mesh {
        Triangle* triangles{};
        Bounds* nodes{};
        int* order{};
        int triangle_count{}, leaf_count{};
        Bounds bounds{};
        float band{};
    };

    struct Motion {
        float matrix[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        float derivative[12]{};
        float time{}, max_speed{};
        Bounds swept{};
    };

    struct Source {
        float3 center{};
        float3 half_extent{0.25f, 0.25f, 0.25f};
        float3 velocity{};
        float smoke_rate{1.0f};
        float temperature_rate{1.0f};
        float velocity_rate{4.0f};
        float start_time{};
        float end_time{1.0e20f};
        Shape shape{Shape::sphere};
        Mesh mesh;
        Motion motion;
        float inherit_velocity{};
    };

    struct Collider {
        float3 center{};
        float3 half_extent{0.5f, 0.5f, 0.5f};
        float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
        float3 linear_velocity{};
        float3 angular_velocity{};
        float pose_time{};
        Shape shape{Shape::sphere};
        Mesh mesh;
        Motion motion;
        float surface_offset{};
    };

    void build_mesh(Mesh mesh, cudaStream_t stream);
#ifdef __CUDACC__
    __device__ inline float3 transform(const float* m, float3 p) {
        return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7], m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
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
    __device__ inline float mesh_distance(Mesh mesh, Motion motion, float3 world, float time, float3* velocity = nullptr, float offset = 0, float* divergence = nullptr, float query_band = 0) {
        float band = (query_band > 0 ? query_band : mesh.band) + fabsf(offset);
        if (velocity) *velocity = {};
        if (divergence) *divergence = 0;
        if (world.x < motion.swept.lower.x - band || world.y < motion.swept.lower.y - band || world.z < motion.swept.lower.z - band || world.x > motion.swept.upper.x + band || world.y > motion.swept.upper.y + band || world.z > motion.swept.upper.z + band) return band - offset;
        float m[12];
        for (int i = 0; i < 12; ++i) m[i] = motion.matrix[i] + (time - motion.time) * motion.derivative[i];
        float3 a{m[0], m[4], m[8]}, b{m[1], m[5], m[9]}, c{m[2], m[6], m[10]};
        float3 u{b.y * c.z - b.z * c.y, b.z * c.x - b.x * c.z, b.x * c.y - b.y * c.x};
        float3 v{c.y * a.z - c.z * a.y, c.z * a.x - c.x * a.z, c.x * a.y - c.y * a.x};
        float3 w{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        float inverse_det = 1.0f / (a.x * u.x + a.y * u.y + a.z * u.z);
        u                 = {u.x * inverse_det, u.y * inverse_det, u.z * inverse_det};
        v                 = {v.x * inverse_det, v.y * inverse_det, v.z * inverse_det};
        w                 = {w.x * inverse_det, w.y * inverse_det, w.z * inverse_det};
        float3 q{world.x - m[3], world.y - m[7], world.z - m[11]};
        float3 p{u.x * q.x + u.y * q.y + u.z * q.z, v.x * q.x + v.y * q.y + v.z * q.z, w.x * q.x + w.y * q.y + w.z * q.z};
        if (velocity) *velocity = transform(motion.derivative, p);
        if (divergence) *divergence = (motion.derivative[0] * u.x + motion.derivative[4] * u.y + motion.derivative[8] * u.z) + (motion.derivative[1] * v.x + motion.derivative[5] * v.y + motion.derivative[9] * v.z) + (motion.derivative[2] * w.x + motion.derivative[6] * w.y + motion.derivative[10] * w.z);
        float metric       = 1.0f / ((u.x * u.x + u.y * u.y + u.z * u.z) + (v.x * v.x + v.y * v.y + v.z * v.z) + (w.x * w.x + w.y * w.y + w.z * w.z));
        float best         = band * band;
        bool inside_bounds = p.x >= mesh.bounds.lower.x && p.y >= mesh.bounds.lower.y && p.z >= mesh.bounds.lower.z && p.x <= mesh.bounds.upper.x && p.y <= mesh.bounds.upper.y && p.z <= mesh.bounds.upper.z;
        for (int node = 1; node;) {
            Bounds bounds = mesh.nodes[node];
            float3 delta{fmaxf(bounds.lower.x - p.x, fmaxf(p.x - bounds.upper.x, 0.0f)), fmaxf(bounds.lower.y - p.y, fmaxf(p.y - bounds.upper.y, 0.0f)), fmaxf(bounds.lower.z - p.z, fmaxf(p.z - bounds.upper.z, 0.0f))};
            if ((delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) * metric >= best) {
                node = next_node(node);
                continue;
            }
            if (node < mesh.leaf_count) {
                node *= 2;
                continue;
            }
            int leaf = node - mesh.leaf_count;
            if (leaf < mesh.triangle_count) {
                Triangle t = mesh.triangles[mesh.order[leaf]];
                best       = fminf(best, triangle_distance_squared(world, transform(m, t.a), transform(m, t.b), transform(m, t.c)));
            }
            node = next_node(node);
        }
        int crossings = 0;
        if (inside_bounds) {
            constexpr double dy = 0.3713906763541037, dz = 0.1278312937608921;
            for (int node = 1; node;) {
                Bounds bounds = mesh.nodes[node];
                double near   = fmax(0.0, fmax(static_cast<double>(bounds.lower.x - p.x), fmax((bounds.lower.y - p.y) / dy, (bounds.lower.z - p.z) / dz)));
                double far    = fmin(static_cast<double>(bounds.upper.x - p.x), fmin((bounds.upper.y - p.y) / dy, (bounds.upper.z - p.z) / dz));
                if (near > far) {
                    node = next_node(node);
                    continue;
                }
                if (node < mesh.leaf_count) {
                    node *= 2;
                    continue;
                }
                int leaf = node - mesh.leaf_count;
                if (leaf < mesh.triangle_count) {
                    Triangle t = mesh.triangles[mesh.order[leaf]];
                    double ex = double(t.b.x) - t.a.x, ey = double(t.b.y) - t.a.y, ez = double(t.b.z) - t.a.z;
                    double fx = double(t.c.x) - t.a.x, fy = double(t.c.y) - t.a.y, fz = double(t.c.z) - t.a.z;
                    double hx = dy * fz - dz * fy, hy = dz * fx - fz, hz = fy - dy * fx;
                    double denominator = ex * hx + ey * hy + ez * hz;
                    if (denominator != 0) {
                        double sx = double(p.x) - t.a.x, sy = double(p.y) - t.a.y, sz = double(p.z) - t.a.z;
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
        return (crossings & 1 ? -sqrtf(best) : sqrtf(best)) - offset;
    }

    __device__ inline float4 rotation(const Collider& body, float time) {
        float speed  = sqrtf(body.angular_velocity.x * body.angular_velocity.x + body.angular_velocity.y * body.angular_velocity.y + body.angular_velocity.z * body.angular_velocity.z);
        float angle  = 0.5f * speed * (time - body.pose_time);
        float factor = speed > 0.0f ? sinf(angle) / speed : 0.0f;
        float3 a{body.angular_velocity.x * factor, body.angular_velocity.y * factor, body.angular_velocity.z * factor};
        float4 q = body.orientation;
        float3 b{q.x, q.y, q.z};
        float cosine = cosf(angle);
        float3 v{b.x * cosine + a.x * q.w + (a.y * b.z - a.z * b.y), b.y * cosine + a.y * q.w + (a.z * b.x - a.x * b.z), b.z * cosine + a.z * q.w + (a.x * b.y - a.y * b.x)};
        return {v.x, v.y, v.z, cosine * q.w - (a.x * b.x + a.y * b.y + a.z * b.z)};
    }
    __device__ inline float3 rotate(float4 q, float3 p) {
        float3 inner{q.y * p.z - q.z * p.y + p.x * q.w, q.z * p.x - q.x * p.z + p.y * q.w, q.x * p.y - q.y * p.x + p.z * q.w};
        return {p.x + (q.y * inner.z - q.z * inner.y) * 2.0f, p.y + (q.z * inner.x - q.x * inner.z) * 2.0f, p.z + (q.x * inner.y - q.y * inner.x) * 2.0f};
    }
    __device__ inline float solid_distance(float3 world, float time, const Collider* bodies, int count, float3* velocity = nullptr, float* divergence = nullptr, float query_band = 0) {
        float distance = 1.0e20f;
        float3 speed{};
        float rate = 0;
        for (int i = 0; i < count; ++i) {
            Collider body  = bodies[i];
            float interval = time - body.pose_time;
            float3 center{body.center.x + body.linear_velocity.x * interval, body.center.y + body.linear_velocity.y * interval, body.center.z + body.linear_velocity.z * interval};
            float4 q     = rotation(body, time);
            float3 local = rotate({-q.x, -q.y, -q.z, q.w}, {world.x - center.x, world.y - center.y, world.z - center.z});
            float d;
            float3 mesh_velocity{};
            float mesh_divergence = 0;
            if (body.shape == Shape::sphere) d = sqrtf(local.x * local.x + local.y * local.y + local.z * local.z) - body.half_extent.x;
            else if (body.shape == Shape::box) {
                float3 e{fabsf(local.x) - body.half_extent.x, fabsf(local.y) - body.half_extent.y, fabsf(local.z) - body.half_extent.z};
                float3 positive{fmaxf(e.x, 0.0f), fmaxf(e.y, 0.0f), fmaxf(e.z, 0.0f)};
                d = sqrtf(positive.x * positive.x + positive.y * positive.y + positive.z * positive.z) + fminf(fmaxf(e.x, fmaxf(e.y, e.z)), 0.0f);
            } else d = mesh_distance(body.mesh, body.motion, world, time, &mesh_velocity, body.surface_offset, &mesh_divergence, query_band);
            if (d < distance) {
                distance = d;
                rate     = mesh_divergence;
                if (body.shape == Shape::mesh) speed = mesh_velocity;
                else {
                    float3 delta{world.x - center.x, world.y - center.y, world.z - center.z};
                    speed = {body.linear_velocity.x + (body.angular_velocity.y * delta.z - body.angular_velocity.z * delta.y), body.linear_velocity.y + (body.angular_velocity.z * delta.x - body.angular_velocity.x * delta.z), body.linear_velocity.z + (body.angular_velocity.x * delta.y - body.angular_velocity.y * delta.x)};
                }
            }
        }
        if (velocity) *velocity = speed;
        if (divergence) *divergence = rate;
        return distance;
    }
    __device__ inline float triangle_fraction(float a, float b, float c) {
        if (a <= 0.0f && b <= 0.0f && c <= 0.0f) return 0.0f;
        if (a >= 0.0f && b >= 0.0f && c >= 0.0f) return 1.0f;
        if ((a > 0.0f) == (b > 0.0f)) {
            float t = a;
            a       = c;
            c       = t;
        } else if ((a > 0.0f) == (c > 0.0f)) {
            float t = a;
            a       = b;
            b       = t;
        }
        float portion = a * a / ((a - b) * (a - c));
        return a > 0.0f ? portion : 1.0f - portion;
    }

    __device__ inline float3 solid_face_moment(float a, float b, float c, float3 x, float3 y, float3 z) {
        if (a >= 0 && b >= 0 && c >= 0) return {};
        float3 total{(x.x + y.x + z.x) * (1.0f / 3.0f), (x.y + y.y + z.y) * (1.0f / 3.0f), (x.z + y.z + z.z) * (1.0f / 3.0f)};
        if (a <= 0 && b <= 0 && c <= 0) return total;
        if ((a > 0) == (b > 0)) {
            float value  = a;
            a            = c;
            c            = value;
            float3 point = x;
            x            = z;
            z            = point;
        } else if ((a > 0) == (c > 0)) {
            float value  = a;
            a            = b;
            b            = value;
            float3 point = x;
            x            = y;
            y            = point;
        }
        float ab = a / (a - b), ac = a / (a - c);
        float weight = ab * ac / 3.0f;
        float3 moment{(x.x + (x.x + (y.x - x.x) * ab) + (x.x + (z.x - x.x) * ac)) * weight, (x.y + (x.y + (y.y - x.y) * ab) + (x.y + (z.y - x.y) * ac)) * weight, (x.z + (x.z + (y.z - x.z) * ab) + (x.z + (z.z - x.z) * ac)) * weight};
        return a > 0 ? float3{total.x - moment.x, total.y - moment.y, total.z - moment.z} : moment;
    }

    __device__ inline float tetrahedron_fraction(float a, float b, float c, float d) {
        float positive[4], negative[4];
        int n = 0, m = 0;
        for (float v : {a, b, c, d}) {
            if (v > 0) positive[n++] = v;
            else negative[m++] = v;
        }
        if (!n) return 0;
        if (n == 4) return 1;
        if (n == 1) {
            float p = positive[0];
            return p * p * p / ((p - negative[0]) * (p - negative[1]) * (p - negative[2]));
        }
        if (n == 3) {
            float p = negative[0];
            return 1 - p * p * p / ((p - positive[0]) * (p - positive[1]) * (p - positive[2]));
        }
        float p = positive[0], q = positive[1], r = negative[0], s = negative[1];
        return (p * p * q * q - (r + s) * p * q * (p + q) + r * s * (p * p + p * q + q * q)) / ((p - r) * (p - s) * (q - r) * (q - s));
    }

    __device__ inline float source_weight(Source source, float3 p, float h, float time, float3* motion_velocity = nullptr) {
        float3 local{p.x - source.center.x, p.y - source.center.y, p.z - source.center.z};
        float distance;
        if (source.shape == Shape::sphere) distance = sqrtf(local.x * local.x + local.y * local.y + local.z * local.z) - source.half_extent.x;
        else if (source.shape == Shape::mesh) distance = mesh_distance(source.mesh, source.motion, p, time, motion_velocity);
        else {
            float3 q{fabsf(local.x) - source.half_extent.x, fabsf(local.y) - source.half_extent.y, fabsf(local.z) - source.half_extent.z};
            float3 positive{fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f)};
            distance = sqrtf(positive.x * positive.x + positive.y * positive.y + positive.z * positive.z) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
        }
        return fminf(1.0f, fmaxf(0.5f - distance / h, 0.0f));
    }

#endif
} // namespace pyro::cuda
#endif
