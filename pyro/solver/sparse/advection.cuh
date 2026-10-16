#ifndef PYRO_SOLVER_SPARSE_ADVECTION_CUH
#define PYRO_SOLVER_SPARSE_ADVECTION_CUH
#include "grid.cuh"

namespace pyro::sparse {
    void advect(Grid grid, Fields destination, Fields carrier, float cell_size, float3 wind, float time, float dt, ::cuda::stream_ref stream);
}
#endif
