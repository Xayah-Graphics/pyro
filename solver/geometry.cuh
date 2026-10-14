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
        Bounds* nodes{}; // Complete binary Morton tree; root 1, leaves at leafCount.
        int* order{};
        int triangleCount{}, leafCount{};
        Bounds bounds{};
        float band{};
    };

    struct Motion {
        float matrix[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        float derivative[12]{};
        float time{}, maxSpeed{};
        Bounds swept{};
    };

    struct Source {
        float3 center{};
        float3 halfExtent{0.25f, 0.25f, 0.25f};
        float3 velocity{};
        float smokeRate{1.0f}; // Concentration injected per second.
        float temperatureRate{1.0f};
        float velocityRate{4.0f};
        float startTime{};
        float endTime{1.0e20f};
        Shape shape{Shape::sphere};
        Mesh mesh;
        Motion motion;
        float inheritVelocity{};
    };

    struct Collider {
        float3 center{};
        float3 halfExtent{0.5f, 0.5f, 0.5f};
        float4 orientation{0.0f, 0.0f, 0.0f, 1.0f}; // Unit quaternion (x, y, z, w) at poseTime.
        float3 linearVelocity{};
        float3 angularVelocity{};
        float poseTime{};
        Shape shape{Shape::sphere};
        Mesh mesh;
        Motion motion;
        float surfaceOffset{};
    };

    inline constexpr int threads = 256;
    void build_mesh(Mesh mesh, cudaStream_t stream);
#ifdef __CUDACC__
    inline void cuda_check(cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    __host__ __device__ inline float3 operator+(float3 a, float3 b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    __host__ __device__ inline float3 operator-(float3 a, float3 b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    __host__ __device__ inline float3 operator*(float3 a, float b) {
        return {a.x * b, a.y * b, a.z * b};
    }
    __host__ __device__ inline float3 operator/(float3 a, float b) {
        return a * (1.0f / b);
    }
    __host__ __device__ inline float dot(float3 a, float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    __host__ __device__ inline float3 cross(float3 a, float3 b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }
    __host__ __device__ inline float length(float3 a) {
        return sqrtf(dot(a, a));
    }
    __host__ __device__ inline float component(float3 a, int axis) {
        return axis == 0 ? a.x : axis == 1 ? a.y : a.z;
    }
    __host__ __device__ inline int3 offset(int3 a, int axis, int amount) {
        if (axis == 0) a.x += amount;
        if (axis == 1) a.y += amount;
        if (axis == 2) a.z += amount;
        return a;
    }
    __device__ inline float3 transform(const float* m, float3 p) {
        return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7], m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
    }
    __device__ inline int next_node(int node) {
        while (node & 1) node >>= 1;
        return node ? node + 1 : 0;
    }
    __device__ inline float triangle_distance_squared(float3 p, float3 a, float3 b, float3 c) {
        float3 ab = b - a, ac = c - a, ap = p - a;
        float d1 = dot(ab, ap), d2 = dot(ac, ap);
        if (d1 <= 0 && d2 <= 0) return dot(ap, ap);
        float3 bp = p - b;
        float d3 = dot(ab, bp), d4 = dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return dot(bp, bp);
        float vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            float3 v = ap - ab * (d1 / (d1 - d3));
            return dot(v, v);
        }
        float3 cp = p - c;
        float d5 = dot(ab, cp), d6 = dot(ac, cp);
        if (d6 >= 0 && d5 <= d6) return dot(cp, cp);
        float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            float3 v = ap - ac * (d2 / (d2 - d6));
            return dot(v, v);
        }
        float va = d3 * d6 - d5 * d4;
        if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
            float3 v = bp - (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
            return dot(v, v);
        }
        float3 v = ap - ab * (vb / (va + vb + vc)) - ac * (vc / (va + vb + vc));
        return dot(v, v);
    }
    __device__ inline float mesh_distance(Mesh mesh, Motion motion, float3 world, float time, float3* velocity = nullptr, float offset = 0, float* divergence = nullptr, float queryBand = 0) {
        float band = (queryBand > 0 ? queryBand : mesh.band) + fabsf(offset);
        if (velocity) *velocity = {};
        if (divergence) *divergence = 0;
        if (world.x < motion.swept.lower.x - band || world.y < motion.swept.lower.y - band || world.z < motion.swept.lower.z - band || world.x > motion.swept.upper.x + band || world.y > motion.swept.upper.y + band || world.z > motion.swept.upper.z + band) return band - offset;
        float m[12];
        for (int i = 0; i < 12; ++i) m[i] = motion.matrix[i] + (time - motion.time) * motion.derivative[i];
        float3 a{m[0], m[4], m[8]}, b{m[1], m[5], m[9]}, c{m[2], m[6], m[10]};
        float3 u = cross(b, c), v = cross(c, a), w = cross(a, b);
        float det = dot(a, u);
        u         = u / det;
        v         = v / det;
        w         = w / det;
        float3 q  = world - float3{m[3], m[7], m[11]};
        float3 p{dot(u, q), dot(v, q), dot(w, q)};
        if (velocity) *velocity = transform(motion.derivative, p);
        if (divergence) *divergence = dot(float3{motion.derivative[0], motion.derivative[4], motion.derivative[8]}, u) + dot(float3{motion.derivative[1], motion.derivative[5], motion.derivative[9]}, v) + dot(float3{motion.derivative[2], motion.derivative[6], motion.derivative[10]}, w);
        // Frobenius norm bounds the inverse operator norm, including nonuniform scale/shear.
        float metric      = 1.0f / (dot(u, u) + dot(v, v) + dot(w, w));
        float best        = band * band;
        bool insideBounds = p.x >= mesh.bounds.lower.x && p.y >= mesh.bounds.lower.y && p.z >= mesh.bounds.lower.z && p.x <= mesh.bounds.upper.x && p.y <= mesh.bounds.upper.y && p.z <= mesh.bounds.upper.z;
        for (int node = 1; node;) {
            Bounds bounds = mesh.nodes[node];
            float3 delta{fmaxf(bounds.lower.x - p.x, fmaxf(p.x - bounds.upper.x, 0.0f)), fmaxf(bounds.lower.y - p.y, fmaxf(p.y - bounds.upper.y, 0.0f)), fmaxf(bounds.lower.z - p.z, fmaxf(p.z - bounds.upper.z, 0.0f))};
            if (dot(delta, delta) * metric >= best) {
                node = next_node(node);
                continue;
            }
            if (node < mesh.leafCount) {
                node *= 2;
                continue;
            }
            int leaf = node - mesh.leafCount;
            if (leaf < mesh.triangleCount) {
                Triangle t = mesh.triangles[mesh.order[leaf]];
                best       = fminf(best, triangle_distance_squared(world, transform(m, t.a), transform(m, t.b), transform(m, t.c)));
            }
            node = next_node(node);
        }
        int crossings = 0;
        if (insideBounds) {
            // A fixed oblique ray avoids axis-aligned grid/triangle coincidences.
            constexpr double dy = 0.3713906763541037, dz = 0.1278312937608921;
            for (int node = 1; node;) {
                Bounds bounds = mesh.nodes[node];
                double near   = fmax(0.0, fmax(static_cast<double>(bounds.lower.x - p.x), fmax((bounds.lower.y - p.y) / dy, (bounds.lower.z - p.z) / dz)));
                double far    = fmin(static_cast<double>(bounds.upper.x - p.x), fmin((bounds.upper.y - p.y) / dy, (bounds.upper.z - p.z) / dz));
                if (near > far) {
                    node = next_node(node);
                    continue;
                }
                if (node < mesh.leafCount) {
                    node *= 2;
                    continue;
                }
                int leaf = node - mesh.leafCount;
                if (leaf < mesh.triangleCount) {
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
        float speed  = length(body.angularVelocity);
        float angle  = 0.5f * speed * (time - body.poseTime);
        float factor = speed > 0.0f ? sinf(angle) / speed : 0.0f;
        float3 a     = body.angularVelocity * factor;
        float4 q     = body.orientation;
        float3 b{q.x, q.y, q.z};
        float3 v = b * cosf(angle) + a * q.w + cross(a, b);
        return {v.x, v.y, v.z, cosf(angle) * q.w - dot(a, b)};
    }
    __device__ inline float3 rotate(float4 q, float3 p) {
        float3 v{q.x, q.y, q.z};
        return p + cross(v, cross(v, p) + p * q.w) * 2.0f;
    }
    __device__ inline float solid_distance(float3 world, float time, const Collider* bodies, int count, float3* velocity = nullptr, float* divergence = nullptr, float queryBand = 0) {
        float distance = 1.0e20f;
        float3 speed{};
        float rate = 0;
        for (int i = 0; i < count; ++i) {
            Collider body = bodies[i];
            float3 center = body.center + body.linearVelocity * (time - body.poseTime);
            float4 q      = rotation(body, time);
            float3 local  = rotate({-q.x, -q.y, -q.z, q.w}, world - center);
            float d;
            float3 meshVelocity{};
            float meshDivergence = 0;
            if (body.shape == Shape::sphere) d = length(local) - body.halfExtent.x;
            else if (body.shape == Shape::box) {
                float3 e{fabsf(local.x) - body.halfExtent.x, fabsf(local.y) - body.halfExtent.y, fabsf(local.z) - body.halfExtent.z};
                d = length({fmaxf(e.x, 0.0f), fmaxf(e.y, 0.0f), fmaxf(e.z, 0.0f)}) + fminf(fmaxf(e.x, fmaxf(e.y, e.z)), 0.0f);
            } else d = mesh_distance(body.mesh, body.motion, world, time, &meshVelocity, body.surfaceOffset, &meshDivergence, queryBand);
            if (d < distance) {
                distance = d;
                rate     = meshDivergence;
                if (body.shape == Shape::mesh) speed = meshVelocity;
                else speed = body.linearVelocity + cross(body.angularVelocity, world - center);
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
        float3 total = (x + y + z) / 3.0f;
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
        float3 moment = (x + (x + (y - x) * ab) + (x + (z - x) * ac)) * (ab * ac / 3.0f);
        return a > 0 ? total - moment : moment;
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

    __device__ inline float source_weight(Source source, float3 p, float h, float time, float3* motionVelocity = nullptr) {
        float3 local = p - source.center;
        float distance;
        if (source.shape == Shape::sphere) distance = length(local) - source.halfExtent.x;
        else if (source.shape == Shape::mesh) distance = mesh_distance(source.mesh, source.motion, p, time, motionVelocity);
        else {
            float3 q{fabsf(local.x) - source.halfExtent.x, fabsf(local.y) - source.halfExtent.y, fabsf(local.z) - source.halfExtent.z};
            distance = length({fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f)}) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
        }
        return fminf(1.0f, fmaxf(0.5f - distance / h, 0.0f));
    }

#endif
} // namespace pyro::cuda
#endif
