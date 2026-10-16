module;
#include "geometry.cuh"
module pyro.geometry;
import std;

namespace pyro {
    cuda::Mesh upload_mesh(std::span<const Triangle> triangles, float band, cudaStream_t stream) {
        static_assert(sizeof(Triangle) == sizeof(cuda::Triangle));
        cuda::Mesh mesh;
        mesh.triangle_count = static_cast<int>(triangles.size());
        mesh.leaf_count     = static_cast<int>(std::bit_ceil(triangles.size()));
        mesh.band          = band;
        mesh.bounds        = {{1.e30f, 1.e30f, 1.e30f}, {-1.e30f, -1.e30f, -1.e30f}};
        for (auto triangle : triangles)
            for (auto p : {triangle.a, triangle.b, triangle.c}) {
                mesh.bounds.lower = {std::min(mesh.bounds.lower.x, p[0]), std::min(mesh.bounds.lower.y, p[1]), std::min(mesh.bounds.lower.z, p[2])};
                mesh.bounds.upper = {std::max(mesh.bounds.upper.x, p[0]), std::max(mesh.bounds.upper.y, p[1]), std::max(mesh.bounds.upper.z, p[2])};
            }
        try {
            if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&mesh.triangles), triangles.size_bytes(), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&mesh.order), triangles.size() * sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMallocAsync(reinterpret_cast<void**>(&mesh.nodes), mesh.leaf_count * 2ull * sizeof(cuda::Bounds), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMemcpyAsync(mesh.triangles, triangles.data(), triangles.size_bytes(), cudaMemcpyHostToDevice, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            cuda::build_mesh(mesh, stream);
            if (auto cuda_result = cudaGetLastError(); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        } catch (...) {
            if (mesh.triangles) cudaFreeAsync(mesh.triangles, stream);
            if (mesh.order) cudaFreeAsync(mesh.order, stream);
            if (mesh.nodes) cudaFreeAsync(mesh.nodes, stream);
            throw;
        }
        return mesh;
    }
}
