module;
#include "geometry.cuh"
export module pyro.geometry;
export import pyro.types;
import std;

export namespace pyro {
    cuda::Mesh upload_mesh(std::span<const Triangle> triangles, float band, cudaStream_t stream);
}
