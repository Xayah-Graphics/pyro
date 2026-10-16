#ifndef PYRO_SOLVER_SPARSE_STATISTICS_CUH
#define PYRO_SOLVER_SPARSE_STATISTICS_CUH

#include "forces.cuh"
#include "grid.cuh"

namespace pyro {
    struct SourceView;
}

namespace pyro::sparse {
    struct Statistics {
        int blocks{};
        int capacity{};
        int fluid_cells{};
        int substeps{};
        int pressure_iterations{};
        int pressure_levels{};

        float max_divergence{};
        float max_speed{};
        float wall_milliseconds{};

        float topology_milliseconds{};
        float geometry_milliseconds{};
        float advection_milliseconds{};
        float forces_milliseconds{};
        float projection_milliseconds{};
        float statistics_milliseconds{};

        double smoke_integral{};
        double temperature_integral{};

        float gpu_milliseconds{};
        float input_milliseconds{};

        unsigned long long allocated_bytes{};
        unsigned long long reserved_bytes{};
    };

    void measure(Grid grid, ForcesConfig config, const SourceView* sources, int source_count, float time, bool full, double* reductions, ::cuda::stream_ref stream);
}
#endif
