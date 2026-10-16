#ifndef PYRO_GEOMETRY_BVH_CUH
#define PYRO_GEOMETRY_BVH_CUH
#include "geometry.cuh"

namespace geometry {
    void build_bvh(MeshView mesh, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool);
}
#endif
