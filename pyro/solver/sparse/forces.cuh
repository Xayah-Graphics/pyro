#ifndef PYRO_SOLVER_SPARSE_FORCES_CUH
#define PYRO_SOLVER_SPARSE_FORCES_CUH

#include "grid.cuh"

namespace pyro {
    struct SourceView;
}

namespace pyro::sparse {
    struct ForcesConfig {
        float buoyancy{1};
        float smoke_weight{0.05f};
        float3 up{0, 1, 0};
        float cooling{0.1f};
        float dissipation{0.01f};
    };

    void sample_sources(Grid grid, const SourceView* sources, int count, float time, float* weights, ::cuda::stream_ref stream);

    void forces(Grid grid, Fields destination, Fields carrier, ForcesConfig config, const SourceView* sources, std::span<const Source> inputs, const float* weights, float time, float dt, ::cuda::stream_ref stream);
}
#endif
