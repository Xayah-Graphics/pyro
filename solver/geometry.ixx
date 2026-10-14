module;
#include "geometry.cuh"
export module pyro.geometry;
export import pyro.types;
import std;

export namespace pyro {
    void cuda_check(cudaError_t result);
    cuda::Source device_input(const Source& source, std::span<const cuda::Mesh> meshes);
    cuda::Collider device_input(const Collider& collider, std::span<const cuda::Mesh> meshes);
    bool source_geometry_changed(const Source& current, const Source& previous);
    bool sources_changed(std::span<const Source> current, std::span<const Source> previous);
    float collider_speed(std::span<const Collider> colliders);
    cuda::Mesh upload_mesh(std::span<const Triangle> triangles, float band, cudaStream_t stream);
} // namespace pyro
