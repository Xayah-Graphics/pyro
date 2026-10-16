#ifndef PYRO_SOLVER_DENSE_FORCES_CUH
#define PYRO_SOLVER_DENSE_FORCES_CUH

#include "grid.cuh"

namespace pyro {
    struct SourceView;
}

namespace pyro::dense {
    struct ForcesConfig {
        float buoyancy{1};
        float smoke_weight{0.05f};
        float3 up{0, 1, 0};
        float cooling{0.1f};
        float dissipation{0.01f};
    };

    struct SourceRegion {
        int3 lower{};
        int3 extent{};
        float* weights{};
        bool cached{};
    };

    void cache_source(Grid grid, float cell_size, const SourceView& source, SourceRegion region, ::cuda::stream_ref stream);

    void apply_sources_forces(Grid grid, ForcesConfig config, float cell_size, float3 wind, Fields destination, Fields force_fields, const SourceRegion* regions, std::span<const Source> inputs, std::span<const ::geometry::InstanceView> instances, float time, float dt, ::cuda::stream_ref stream);
}
#endif
