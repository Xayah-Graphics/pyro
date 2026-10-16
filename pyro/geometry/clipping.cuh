#ifndef GEOMETRY_CLIPPING_CUH
#define GEOMETRY_CLIPPING_CUH

#include <initializer_list>

#include <cuda_runtime.h>

namespace geometry {
    __device__ inline float triangle_positive_fraction(float a, float b, float c) {
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

    __device__ inline float3 triangle_negative_moment(float a, float b, float c, float3 x, float3 y, float3 z) {
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

    __device__ inline float tetrahedron_positive_fraction(float a, float b, float c, float d) {
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
}
#endif
