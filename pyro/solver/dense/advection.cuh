#ifndef PYRO_SOLVER_DENSE_ADVECTION_CUH
#define PYRO_SOLVER_DENSE_ADVECTION_CUH
#include "grid.cuh"

namespace pyro::dense {
    void advect(Grid grid, float cell_size, float3 wind, Fields destination, Fields original, Fields carrier, float start_time, float dt, ::cuda::stream_ref stream);
}
#endif
