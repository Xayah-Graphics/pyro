#include "bvh.cuh"

#include <cstddef>
#include <stdexcept>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/buffer>
#include <cuda/launch>

namespace geometry {
    __device__ unsigned morton_bits(unsigned v) {
        v = (v | (v << 16)) & 0x030000ffu;
        v = (v | (v << 8)) & 0x0300f00fu;
        v = (v | (v << 4)) & 0x030c30c3u;
        return (v | (v << 2)) & 0x09249249u;
    }

    __global__ void mesh_keys(MeshView mesh, BvhView bvh, unsigned* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= mesh.triangle_count) return;
        int3 triangle = mesh.triangles[i];
        float3 ta = mesh.positions[triangle.x], tb = mesh.positions[triangle.y], tc = mesh.positions[triangle.z];
        float3 p{(ta.x + tb.x + tc.x) * (1.0f / 3.0f), (ta.y + tb.y + tc.y) * (1.0f / 3.0f), (ta.z + tb.z + tc.z) * (1.0f / 3.0f)};
        float3 extent{mesh.bounds.upper.x - mesh.bounds.lower.x, mesh.bounds.upper.y - mesh.bounds.lower.y, mesh.bounds.upper.z - mesh.bounds.lower.z};
        unsigned x   = min(1023u, static_cast<unsigned>(1023.0f * (p.x - mesh.bounds.lower.x) / fmaxf(extent.x, 1.0e-20f)));
        unsigned y   = min(1023u, static_cast<unsigned>(1023.0f * (p.y - mesh.bounds.lower.y) / fmaxf(extent.y, 1.0e-20f)));
        unsigned z   = min(1023u, static_cast<unsigned>(1023.0f * (p.z - mesh.bounds.lower.z) / fmaxf(extent.z, 1.0e-20f)));
        keys[i]      = morton_bits(x) | (morton_bits(y) << 1) | (morton_bits(z) << 2);
        bvh.order[i] = i;
    }

    __global__ void mesh_leaves(MeshView mesh, BvhView bvh) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= bvh.leaf_count) return;
        Bounds b{{1.0e30f, 1.0e30f, 1.0e30f}, {-1.0e30f, -1.0e30f, -1.0e30f}};
        if (i < mesh.triangle_count) {
            int3 triangle = mesh.triangles[bvh.order[i]];
            float3 ta = mesh.positions[triangle.x], tb = mesh.positions[triangle.y], tc = mesh.positions[triangle.z];
            b.lower = {fminf(ta.x, fminf(tb.x, tc.x)), fminf(ta.y, fminf(tb.y, tc.y)), fminf(ta.z, fminf(tb.z, tc.z))};
            b.upper = {fmaxf(ta.x, fmaxf(tb.x, tc.x)), fmaxf(ta.y, fmaxf(tb.y, tc.y)), fmaxf(ta.z, fmaxf(tb.z, tc.z))};
        }
        bvh.nodes[bvh.leaf_count + i] = b;
    }

    __global__ void mesh_parents(BvhView bvh, int first) {
        int i = first + blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= first * 2) return;
        Bounds a = bvh.nodes[i * 2], b = bvh.nodes[i * 2 + 1];
        bvh.nodes[i] = {{fminf(a.lower.x, b.lower.x), fminf(a.lower.y, b.lower.y), fminf(a.lower.z, b.lower.z)}, {fmaxf(a.upper.x, b.upper.x), fmaxf(a.upper.y, b.upper.y), fmaxf(a.upper.z, b.upper.z)}};
    }

    void build_bvh(MeshView mesh, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool) {
        auto bvh               = mesh.bvh;
        constexpr int threads  = 256;
        auto keys              = ::cuda::make_buffer<unsigned>(stream, pool, mesh.triangle_count * 2ull, ::cuda::no_init);
        auto order             = ::cuda::make_buffer<int>(stream, pool, mesh.triangle_count, ::cuda::no_init);
        std::size_t sort_bytes = 0;
        if (auto cuda_result = cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys.data(), keys.data() + mesh.triangle_count, order.data(), bvh.order, mesh.triangle_count, 0, 30, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        auto sorting   = ::cuda::make_buffer<unsigned char>(stream, pool, sort_bytes, ::cuda::no_init);
        auto unsorted  = bvh;
        unsorted.order = order.data();
        ::cuda::launch(stream, ::cuda::distribute<threads>(mesh.triangle_count), mesh_keys, mesh, unsorted, keys.data());
        if (auto cuda_result = cub::DeviceRadixSort::SortPairs(sorting.data(), sort_bytes, keys.data(), keys.data() + mesh.triangle_count, order.data(), bvh.order, mesh.triangle_count, 0, 30, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        ::cuda::launch(stream, ::cuda::distribute<threads>(bvh.leaf_count), mesh_leaves, mesh, bvh);
        for (int first = bvh.leaf_count / 2; first; first /= 2) ::cuda::launch(stream, ::cuda::distribute<threads>(first), mesh_parents, bvh, first);
    }

}
