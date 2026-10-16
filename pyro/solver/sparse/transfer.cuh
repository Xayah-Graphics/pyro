#ifndef PYRO_SOLVER_SPARSE_TRANSFER_CUH
#define PYRO_SOLVER_SPARSE_TRANSFER_CUH
#include "grid.cuh"

namespace pyro::sparse {
    void remap(Grid old_grid, Grid new_grid, float3 wind, ::cuda::stream_ref stream);
}
#endif
