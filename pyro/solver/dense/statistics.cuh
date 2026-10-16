#ifndef PYRO_SOLVER_DENSE_STATISTICS_CUH
#define PYRO_SOLVER_DENSE_STATISTICS_CUH

#include "forces.cuh"
#include "grid.cuh"

namespace pyro {
    struct SourceView;
}

namespace pyro::dense {
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

    void measure(Grid grid, Fields fields, ForcesConfig config, float cell_size, float3 wind, float max_time_step, const SourceView* sources, int source_count, float time, double* reductions, float* maximum, bool full, ::cuda::stream_ref stream);
}
#endif
