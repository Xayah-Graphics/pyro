#ifndef PYRO_SOLVER_LFM_STATISTICS_CUH
#define PYRO_SOLVER_LFM_STATISTICS_CUH

#include "grid.cuh"

namespace pyro::lfm {
    struct Statistics {
        int fluid_cells{};
        int substeps{};
        int pressure_iterations{};
        int pressure_levels{};

        float max_divergence{};
        float max_speed{};
        float wall_milliseconds{};

        float geometry_milliseconds{};
        float advection_milliseconds{};
        float flow_map_milliseconds{};
        float projection_milliseconds{};
        float statistics_milliseconds{};

        float gpu_milliseconds{};
        float input_milliseconds{};

        unsigned long long allocated_bytes{};
        unsigned long long reserved_bytes{};
    };

    void measure(Grid grid, Velocity velocity, float h, double* reductions, ::cuda::stream_ref stream);

    void export_vorticity(Grid grid, Velocity velocity, float h, ::cuda::stream_ref stream);
}
#endif
