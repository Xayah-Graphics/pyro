#ifndef PYRO_SOLVER_LFM_ADVECTION_CUH
#define PYRO_SOLVER_LFM_ADVECTION_CUH
#include "grid.cuh"

namespace pyro::lfm {
    void advect(Grid grid, Velocity destination, Velocity original, Velocity carrier, float dt_over_h, float3 wind, ::cuda::stream_ref stream);
}
#endif
