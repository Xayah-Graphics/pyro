module;
#include "geometry.cuh"
module pyro.geometry;
import std;

namespace pyro {
    void cuda_check(cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    cuda::Source device_input(const Source& source, std::span<const cuda::Mesh> meshes) {
        return {std::bit_cast<float3>(source.center), std::bit_cast<float3>(source.halfExtent), std::bit_cast<float3>(source.velocity), source.smokeRate, source.temperatureRate, source.velocityRate, source.startTime, source.endTime, static_cast<cuda::Shape>(source.shape), source.shape == Shape::mesh ? meshes[source.mesh] : cuda::Mesh{}, std::bit_cast<cuda::Motion>(source.motion), source.inheritVelocity};
    }

    cuda::Collider device_input(const Collider& collider, std::span<const cuda::Mesh> meshes) {
        return {std::bit_cast<float3>(collider.center), std::bit_cast<float3>(collider.halfExtent), std::bit_cast<float4>(collider.orientation), std::bit_cast<float3>(collider.linearVelocity), std::bit_cast<float3>(collider.angularVelocity), collider.poseTime, static_cast<cuda::Shape>(collider.shape), collider.shape == Shape::mesh ? meshes[collider.mesh] : cuda::Mesh{}, std::bit_cast<cuda::Motion>(collider.motion), collider.surfaceOffset};
    }

    cuda::Mesh upload_mesh(std::span<const Triangle> triangles, float band, cudaStream_t stream) {
        static_assert(sizeof(Triangle) == sizeof(cuda::Triangle));
        cuda::Mesh mesh;
        mesh.triangleCount = static_cast<int>(triangles.size());
        mesh.leafCount     = static_cast<int>(std::bit_ceil(triangles.size()));
        mesh.band          = band;
        mesh.bounds        = {{1.e30f, 1.e30f, 1.e30f}, {-1.e30f, -1.e30f, -1.e30f}};
        for (auto triangle : triangles)
            for (auto p : {triangle.a, triangle.b, triangle.c}) {
                mesh.bounds.lower = {std::min(mesh.bounds.lower.x, p[0]), std::min(mesh.bounds.lower.y, p[1]), std::min(mesh.bounds.lower.z, p[2])};
                mesh.bounds.upper = {std::max(mesh.bounds.upper.x, p[0]), std::max(mesh.bounds.upper.y, p[1]), std::max(mesh.bounds.upper.z, p[2])};
            }
        try {
            cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&mesh.triangles), triangles.size_bytes(), stream));
            cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&mesh.order), triangles.size() * sizeof(int), stream));
            cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&mesh.nodes), mesh.leafCount * 2ull * sizeof(cuda::Bounds), stream));
            cuda_check(cudaMemcpyAsync(mesh.triangles, triangles.data(), triangles.size_bytes(), cudaMemcpyHostToDevice, stream));
            cuda::build_mesh(mesh, stream);
            cuda_check(cudaGetLastError());
            cuda_check(cudaStreamSynchronize(stream));
        } catch (...) {
            if (mesh.triangles) cudaFreeAsync(mesh.triangles, stream);
            if (mesh.order) cudaFreeAsync(mesh.order, stream);
            if (mesh.nodes) cudaFreeAsync(mesh.nodes, stream);
            throw;
        }
        return mesh;
    }
} // namespace pyro
