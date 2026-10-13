#ifndef PYRO_CUDA_CUH
#define PYRO_CUDA_CUH

#include "../geometry.cuh"
#ifdef __CUDACC__
#include <cmath>
#include <stdexcept>
#endif

namespace pyro::cuda {
    struct Config {
        float cellSize{0.05f};
        int maxBlocks{4096};
        int pressurePadding{2}; // In 8-cell blocks; independent of the advection stencil.
        int keepAliveSteps{4};
        float smokeThreshold{1.0e-5f};
        float temperatureThreshold{1.0e-4f};
        float velocityThreshold{0.025f};
        float cfl{0.75f};
        float maxTimeStep{1.0f / 48.0f};
        float buoyancy{1.0f};
        float smokeWeight{0.05f};
        float3 up{0.0f, 1.0f, 0.0f};
        float3 wind{};
        float cooling{0.1f};
        float dissipation{0.01f};
        float pressureRelativeTolerance{1.0e-5f};
        float divergenceTolerance{2.0e-4f}; // Absolute max divergence, in inverse seconds.
        int pressureMaxIterations{200};
        int maxPressureLevels{12};
    };

    struct DensityBlock {
        int3 coordinate{};
        float values[512]; // x contiguous; cell-centered samples.
    };

    // Low bits are storage/domain/new allocation; higher bits explain the last topology request.
    inline constexpr unsigned gridSmoke = 8u, gridTemperature = 16u, gridVelocity = 32u;
    inline constexpr unsigned gridEmitter = 64u, gridCollider = 128u, gridRetained = 256u, gridPadding = 512u;

    struct GridBlock {
        int3 coordinate{};
        unsigned flags;
        int remaining; // Future inactive substeps that will still retain this block as a seed.
    };

    struct StateBlock {
        int id;
        int3 coordinate{};
        unsigned role;
        int age;
        float smoke[512], temperature[512], velocity[3][512];
    };

    struct Fields {
        // Negative MAC faces at (x,y+1/2,z+1/2), etc.; scalars at cell centers.
        float* velocity[3]{};
        float* smoke{};
        float* temperature{};
    };

    struct Layout {
        unsigned long long* keys{};
        int* indices{};
        int3* coordinates{};
        int* active{};
        int* neighbors{};
        int* count{};
        int capacity{};
        int tableCapacity{};
        int blocks{};
    };

    struct Grid {
        Layout layout;
        unsigned long long* nextKeys{};
        int* nextIndices{};
        unsigned* requested{};
        unsigned* roles{};
        int* freeBlocks{};
        int* freeCount{};
        int* age{};
        unsigned char* fluid{};
        unsigned char* boundaryDirty{}; // Per block: allocation, domain role or corner-neighbor changes.
        float* aperture[3]{};
        float* solidVelocity[3]{};
        float* solidDivergence{};
        float* distance[3]{}; // Sparse corner samples at substep start, midpoint and end.
        float geometryTime{}, geometryDuration{};
        bool staticGeometry{true};
        Fields state, midpoint, next, transported;
    };

    struct PressureLevel {
        Layout layout;
        float* diagonal{};
        float* boundary{}; // Dirichlet diagonal, separate from the conservative face sum.
        float* face[3]{}; // Negative faces; each off-diagonal has one owner.
        float* solution{};
        float* rhs{};
        float* temporary{};
        int storageCapacity{};
    };

    struct IterationState {
        double rz{}, previousRz{}, denominator{}, initialNorm{}, residualNorm{};
        float maxResidual{};
        int iterations{}, converged{}, failed{};
    };

    struct PressureWorkspace {
        PressureLevel* levels{};
        int levelCount{};
        float* direction{};
        float* product{};
        float* solution{};
        float* residual{};
        int* parents{};
        int* openComponents{};
        unsigned char* anchors{};
        double* reductions{};
        IterationState* iteration{};
        cudaGraphExec_t graph{};
    };

    inline constexpr int blockWidth              = 8;
    inline constexpr int blockCells              = 512;
    inline constexpr unsigned long long emptyKey = ~0ull;

    void build_neighbors(Layout layout, cudaStream_t stream);
    void request_topology(Grid grid, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, float dt, int* error, cudaStream_t stream);
    void initialize_topology(Grid grid, float3 wind, cudaStream_t stream);
    void sample_geometry(Grid grid, float h, const Collider* colliders, int count, float time, int sample, bool onlyNew, cudaStream_t stream);
    void update_boundary(Grid grid, float h, const Collider* colliders, int count, float time, int sample, bool onlyDirty, cudaStream_t stream);
    void advect(Grid grid, Config config, Fields destination, Fields original, Fields carrier, const Collider* colliders, int count, float startTime, float dt, cudaStream_t stream);
    void apply_sources_forces(Grid grid, Config config, Fields destination, Fields forceFields, const Source* sources, int count, float time, float dt, cudaStream_t stream);
    void measure(Grid grid, Fields fields, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, double* reductions, float* maximum, bool full, cudaStream_t stream);
    void precondition(PressureWorkspace workspace, cudaStream_t stream);
    void assemble_pressure(Grid grid, Fields fields, PressureWorkspace workspace, float h, bool obstacles, bool rebuild, cudaStream_t stream);
    void coarse_topology(Layout fine, Layout coarse, cudaStream_t stream);
    void coarse_matrix(PressureLevel fine, PressureLevel coarse, cudaStream_t stream);
    void begin_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream);
    void iterate_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream);
    void check_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream);
    void apply_pressure(Grid grid, Fields fields, const float* pressure, float h, float3 wind, cudaStream_t stream);
    void pack_density(Grid grid, DensityBlock* destination, int* count, cudaStream_t stream);
    void transfer_state(Grid grid, StateBlock* blocks, bool restore, cudaStream_t stream);
    void pack_grid(Grid grid, GridBlock* blocks, cudaStream_t stream);

#ifdef __CUDACC__
    __host__ __device__ inline unsigned long long key(int3 p) {
        return (static_cast<unsigned long long>(p.x + 1048576) & 0x1fffffull) | ((static_cast<unsigned long long>(p.y + 1048576) & 0x1fffffull) << 21) | ((static_cast<unsigned long long>(p.z + 1048576) & 0x1fffffull) << 42);
    }
    __host__ __device__ inline int3 coordinate(unsigned long long value) {
        return {static_cast<int>(value & 0x1fffffull) - 1048576, static_cast<int>((value >> 21) & 0x1fffffull) - 1048576, static_cast<int>((value >> 42) & 0x1fffffull) - 1048576};
    }
    __device__ inline int hash_slot(unsigned long long value, int capacity) {
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ull;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebull;
        value ^= value >> 31;
        return static_cast<int>(value & (capacity - 1));
    }
    __device__ inline int find_slot(const Layout& layout, int3 p) {
        auto value = key(p);
        int slot   = hash_slot(value, layout.tableCapacity);
        for (int i = 0; i < layout.tableCapacity; ++i) {
            auto present = layout.keys[slot];
            if (present == value) return slot;
            if (present == emptyKey) return -1;
            slot = (slot + 1) & (layout.tableCapacity - 1);
        }
        return -1;
    }
    __device__ inline int find_block(const Layout& layout, int3 p) {
        int slot = find_slot(layout, p);
        return slot < 0 ? -1 : layout.indices[slot];
    }
    __device__ inline int cell_index(const Layout& layout, int3 p) {
        int block = find_block(layout, {p.x >> 3, p.y >> 3, p.z >> 3});
        return block < 0 ? -1 : block * blockCells + (p.x & 7) + 8 * (p.y & 7) + 64 * (p.z & 7);
    }
    __device__ inline int3 cell_coordinate(const Layout& layout, int index) {
        int3 p = layout.coordinates[index >> 9];
        return {p.x * 8 + (index & 7), p.y * 8 + ((index >> 3) & 7), p.z * 8 + ((index >> 6) & 7)};
    }
    __device__ inline int neighbor(const Layout& layout, int index, int axis, int sign) {
        int local = index & 511;
        int shift = axis * 3;
        int v     = (local >> shift) & 7;
        if (v + sign >= 0 && v + sign < 8) return index + sign * (1 << shift);
        int block = layout.neighbors[(index >> 9) * 6 + axis * 2 + (sign > 0)];
        return block < 0 ? -1 : block * 512 + local - (v << shift) + ((sign > 0 ? 0 : 7) << shift);
    }
    __device__ inline float value(const float* field, int index, float background = 0.0f) {
        return index < 0 ? background : field[index];
    }
    __device__ inline float sample(const Layout& layout, const float* field, float3 position, float background, float* lower = nullptr, float* upper = nullptr) {
        int3 p{static_cast<int>(floorf(position.x)), static_cast<int>(floorf(position.y)), static_cast<int>(floorf(position.z))};
        float3 t{position.x - p.x, position.y - p.y, position.z - p.z};
        float result = 0.0f, lo = 1.0e30f, hi = -1.0e30f;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    float v = value(field, cell_index(layout, {p.x + x, p.y + y, p.z + z}), background);
                    result += v * (x ? t.x : 1.0f - t.x) * (y ? t.y : 1.0f - t.y) * (z ? t.z : 1.0f - t.z);
                    lo = fminf(lo, v);
                    hi = fmaxf(hi, v);
                }
        if (lower) *lower = lo;
        if (upper) *upper = hi;
        return result;
    }
    __device__ inline float3 sample_velocity(const Layout& layout, Fields fields, float3 p, float3 wind) {
        return {sample(layout, fields.velocity[0], {p.x, p.y - 0.5f, p.z - 0.5f}, wind.x), sample(layout, fields.velocity[1], {p.x - 0.5f, p.y, p.z - 0.5f}, wind.y), sample(layout, fields.velocity[2], {p.x - 0.5f, p.y - 0.5f, p.z}, wind.z)};
    }

    __device__ inline float sample_boundary(Grid grid, float3 world, float time, float h) {
        if (grid.staticGeometry) return sample(grid.layout, grid.distance[0], world / h, 4.0f * h);
        float fraction = fminf(2.0f, fmaxf(0.0f, 2.0f * (time - grid.geometryTime) / grid.geometryDuration));
        int lower      = fraction < 1.0f ? 0 : 1;
        float weight   = fraction - lower;
        float a        = sample(grid.layout, grid.distance[lower], world / h, 4.0f * h);
        float b        = sample(grid.layout, grid.distance[lower + 1], world / h, 4.0f * h);
        return a + (b - a) * weight;
    }

#endif
} // namespace pyro::cuda

#endif
