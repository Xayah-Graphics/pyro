#ifndef PYRO_SOLVER_LFM_FLOW_MAP_CUH
#define PYRO_SOLVER_LFM_FLOW_MAP_CUH
#include "grid.cuh"

namespace pyro::lfm {
    void reconstruct(Grid grid, int steps, float cell_size, float3 wind, float dt, ::cuda::stream_ref stream);
}
#endif
