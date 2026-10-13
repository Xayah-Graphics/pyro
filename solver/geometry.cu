#include "geometry.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

namespace pyro::cuda {
    __device__ unsigned morton_bits(unsigned v) {
        v = (v | (v << 16)) & 0x030000ffu;
        v = (v | (v << 8)) & 0x0300f00fu;
        v = (v | (v << 4)) & 0x030c30c3u;
        return (v | (v << 2)) & 0x09249249u;
    }

    __global__ void mesh_keys(Mesh mesh, unsigned* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= mesh.triangleCount) return;
        Triangle t    = mesh.triangles[i];
        float3 p      = (t.a + t.b + t.c) / 3.0f;
        float3 extent = mesh.bounds.upper - mesh.bounds.lower;
        unsigned x    = min(1023u, static_cast<unsigned>(1023.0f * (p.x - mesh.bounds.lower.x) / fmaxf(extent.x, 1.0e-20f)));
        unsigned y    = min(1023u, static_cast<unsigned>(1023.0f * (p.y - mesh.bounds.lower.y) / fmaxf(extent.y, 1.0e-20f)));
        unsigned z    = min(1023u, static_cast<unsigned>(1023.0f * (p.z - mesh.bounds.lower.z) / fmaxf(extent.z, 1.0e-20f)));
        keys[i]       = morton_bits(x) | (morton_bits(y) << 1) | (morton_bits(z) << 2);
        mesh.order[i] = i;
    }

    __global__ void mesh_leaves(Mesh mesh) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= mesh.leafCount) return;
        Bounds b{{1.0e30f, 1.0e30f, 1.0e30f}, {-1.0e30f, -1.0e30f, -1.0e30f}};
        if (i < mesh.triangleCount) {
            Triangle t = mesh.triangles[mesh.order[i]];
            b.lower    = {fminf(t.a.x, fminf(t.b.x, t.c.x)), fminf(t.a.y, fminf(t.b.y, t.c.y)), fminf(t.a.z, fminf(t.b.z, t.c.z))};
            b.upper    = {fmaxf(t.a.x, fmaxf(t.b.x, t.c.x)), fmaxf(t.a.y, fmaxf(t.b.y, t.c.y)), fmaxf(t.a.z, fmaxf(t.b.z, t.c.z))};
        }
        mesh.nodes[mesh.leafCount + i] = b;
    }

    __global__ void mesh_parents(Mesh mesh, int first) {
        int i = first + blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= first * 2) return;
        Bounds a = mesh.nodes[i * 2], b = mesh.nodes[i * 2 + 1];
        mesh.nodes[i] = {{fminf(a.lower.x, b.lower.x), fminf(a.lower.y, b.lower.y), fminf(a.lower.z, b.lower.z)}, {fmaxf(a.upper.x, b.upper.x), fmaxf(a.upper.y, b.upper.y), fmaxf(a.upper.z, b.upper.z)}};
    }

    void build_mesh(Mesh mesh, cudaStream_t stream) {
        unsigned* keys;
        cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&keys), mesh.triangleCount * sizeof(unsigned), stream));
        mesh_keys<<<(mesh.triangleCount + threads - 1) / threads, threads, 0, stream>>>(mesh, keys);
        thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(keys), thrust::device_pointer_cast(keys + mesh.triangleCount), thrust::device_pointer_cast(mesh.order));
        mesh_leaves<<<(mesh.leafCount + threads - 1) / threads, threads, 0, stream>>>(mesh);
        for (int first = mesh.leafCount / 2; first; first /= 2) mesh_parents<<<(first + threads - 1) / threads, threads, 0, stream>>>(mesh, first);
        cuda_check(cudaFreeAsync(keys, stream));
    }
} // namespace pyro::cuda
