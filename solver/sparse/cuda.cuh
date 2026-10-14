#ifndef PYRO_SPARSE_CUDA_CUH
#define PYRO_SPARSE_CUDA_CUH
#include "../geometry.cuh"

namespace pyro::cuda::sparse {
    inline constexpr unsigned long long emptyKey = ~0ull;
    struct Config {
        float cellSize{0.05f};
        int initialBlocks{4096};
        int levels{3}, pressurePadding{1}, adaptInterval{2}, coarsenDelay{4}, keepAlive{4};
        float smokeAbsoluteError{0.0003125f}, temperatureAbsoluteError{0.00125f}, velocityAbsoluteError{0.001875f}, relativeError{0.003125f};
        float smokeThreshold{1.e-5f}, temperatureThreshold{1.e-4f}, velocityThreshold{0.025f};
        float cfl{0.75f}, maxTimeStep{1.f / 48}, buoyancy{1}, smokeWeight{0.05f};
        float3 up{0, 1, 0}, wind{};
        float cooling{0.1f}, dissipation{0.01f}, pressureRelativeTolerance{1.e-5f}, divergenceTolerance{2.e-4f};
        int pressureMaxIterations{300};
    };
    struct Node {
        int3 coordinate{};
        int level{};
    };
    struct Block {
        int3 coordinate{};
        int level{}, quiet{}, remaining{};
    };
    struct DensityBlock {
        Block block;
        float values[512];
    };
    struct Fields {
        float* smoke{};
        float* temperature{};
        float* velocity{};
        float* samples{};
    };
    struct Layout {
        Block* blocks{};
        unsigned long long* keys{};
        int* indices{};
        int count{}, capacity{}, tableSize{}, levels{};
        float h{};
    };
    struct Face {
        int left{-1}, right{-1}, group{-1};
        float area{}, solidFlux{}, weight{}, distance{};
    };
    struct Links {
        int face[4]{-1, -1, -1, -1};
    };
    struct Grid {
        Layout layout;
        Face* faces{};
        Links* links{};
        unsigned char* fluid{};
        float* solidDivergence{};
        Fields state, midpoint, next, transported;
        float* samples[3]{}; // Native 11^3 ghost tiles for original, carrier and reverse transport.
        unsigned char* uniformSamples{};
        float* distance[3]{}; // Exact native 9^3 corner samples at t, t+dt/2 and t+dt.
        float geometryTime{}, geometryDuration{};
        bool staticGeometry{true};
    };
    struct Scratch {
        unsigned long long* keys[3]{};
        float* values[2]{};
        int* count{};
        int capacity{};
        void* sortStorage{};
        unsigned long long sortBytes{};
    };
    struct Topology {
        unsigned long long* roots{};
        unsigned long long* refine{};
        int* decision{};
        int* quiet{};
        int* remaining{};
        int* split{};
        Block* candidates{};
        int* counters{};
        int tableSize{}, capacity{};
    };
    struct Edge {
        int left{}, right{};
        float weight{};
    };
    struct PressureLevel {
        Node* nodes{};
        int* parent{};
        int* children{};
        Edge* edges{};
        float* faceWeights{}; // Contiguous finest-level coefficients for structured block interiors.
        int* adjacency{}; // Coarse levels only, at most 24 neighbors on a balanced octree.
        float* boundary{};
        float* diagonal{};
        float* rhs{};
        float* solution{};
        float* temporary{};
        int count{}, edgeCount{}, capacity{}, edgeCapacity{};
    };
    struct IterationState {
        double rz{}, previousRz{}, denominator{}, initialNorm{};
        float maxResidual{};
        int iterations{}, converged{}, failed{};
    };
    struct PressureWorkspace {
        PressureLevel* levels{};
        int levelCount{};
        double* solution{};
        float* residual{};
        float* direction{};
        float* product{};
        int* components{};
        int* open{};
        unsigned char* anchors{};
        double* reductions{};
        double* componentSums{};
        bool closed{};
        IterationState* iteration{};
        cudaGraphExec_t graph{};
    };
    void initialize_scratch(Scratch& scratch, int capacity, cudaStream_t stream);
    void rebuild_hash(Layout layout, cudaStream_t stream);
    // On overflow, return a required capacity larger than newGrid.layout.capacity.
    // The live oldGrid is unchanged, so the caller can reserve and repeat adaptation.
    int adapt(Grid oldGrid, Grid newGrid, Topology topology, Scratch scratch, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, float horizon, bool& changed, cudaStream_t stream);
    void build_connectivity(Grid grid, cudaStream_t stream);
    void sample_geometry(Grid grid, Layout previous, const float* previousDistance, const Collider* colliders, int count, float time, int sample, cudaStream_t stream);
    void geometry(Grid grid, const Collider* colliders, int count, float time, int sample, cudaStream_t stream);
    void remap(Grid oldGrid, Grid newGrid, Config config, cudaStream_t stream);
    void advect(Grid grid, Fields destination, Fields carrier, Config config, const Collider* colliders, int count, float time, float dt, cudaStream_t stream);
    void forces(Grid grid, Fields destination, Fields carrier, Config config, const Source* sources, int count, float time, float dt, cudaStream_t stream);
    void measure(Grid grid, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, bool full, double* reductions, cudaStream_t stream);
    bool fine_matrix(Grid grid, PressureWorkspace workspace, bool obstacles, cudaStream_t stream);
    int coarse_nodes(PressureLevel fine, PressureLevel coarse, Scratch scratch, int depth, cudaStream_t stream);
    int coarse_edges(PressureLevel fine, PressureLevel coarse, Scratch scratch, cudaStream_t stream);
    void assemble_rhs(Grid grid, Fields fields, PressureWorkspace workspace, cudaStream_t stream);
    void begin_pressure(Grid grid, PressureWorkspace workspace, Config config, cudaStream_t stream);
    void iterate_pressure(Grid grid, PressureWorkspace workspace, Config config, cudaStream_t stream);
    void check_pressure(Grid grid, PressureWorkspace workspace, Config config, cudaStream_t stream);
    void apply_pressure(Grid grid, Fields fields, const double* pressure, cudaStream_t stream);
    void pack_density(Grid grid, DensityBlock* output, cudaStream_t stream);

#ifdef __CUDACC__
    __host__ __device__ inline unsigned long long key(int3 p, int level) {
        return (static_cast<unsigned long long>(p.x + 262144) & 0x7ffffull) | ((static_cast<unsigned long long>(p.y + 262144) & 0x7ffffull) << 19) | ((static_cast<unsigned long long>(p.z + 262144) & 0x7ffffull) << 38) | (static_cast<unsigned long long>(level) << 57);
    }
    __host__ __device__ inline Node node(unsigned long long k) {
        return {{int(k & 0x7ffff) - 262144, int((k >> 19) & 0x7ffff) - 262144, int((k >> 38) & 0x7ffff) - 262144}, int(k >> 57)};
    }
    __device__ inline int hash(unsigned long long k, int size) {
        k ^= k >> 30;
        k *= 0xbf58476d1ce4e5b9ull;
        k ^= k >> 27;
        k *= 0x94d049bb133111ebull;
        k ^= k >> 31;
        return int(k & (size - 1));
    }
    __device__ inline int insert(unsigned long long* keys, int size, unsigned long long k) {
        int slot = hash(k, size);
        for (int n = 0; n < size; ++n) {
            auto previous = atomicCAS(keys + slot, emptyKey, k);
            if (previous == emptyKey || previous == k) return slot;
            slot = (slot + 1) & (size - 1);
        }
        return -1;
    }
    __device__ inline int lookup(const unsigned long long* keys, int size, unsigned long long k) {
        int slot = hash(k, size);
        for (int n = 0; n < size; ++n) {
            if (keys[slot] == k) return slot;
            if (keys[slot] == emptyKey) return -1;
            slot = (slot + 1) & (size - 1);
        }
        return -1;
    }
    __device__ inline int find_block(Layout layout, int3 p, int level) {
        int slot = lookup(layout.keys, layout.tableSize, key(p, level));
        return slot < 0 ? -1 : layout.indices[slot];
    }
    __device__ inline int locate(Layout layout, int3 fine) {
        for (int level = 0; level < layout.levels; ++level) {
            int3 c{fine.x >> level, fine.y >> level, fine.z >> level};
            int b = find_block(layout, {c.x >> 3, c.y >> 3, c.z >> 3}, level);
            if (b >= 0) return b * 512 + (c.x & 7) + 8 * (c.y & 7) + 64 * (c.z & 7);
        }
        return -1;
    }
    __device__ inline int locate(Layout layout, float3 p) {
        return locate(layout, int3{int(floorf(p.x / layout.h)), int(floorf(p.y / layout.h)), int(floorf(p.z / layout.h))});
    }
    __device__ inline float spacing(Layout layout, int i) {
        return ldexpf(layout.h, layout.blocks[i >> 9].level);
    }
    __device__ inline int3 cell_coordinate(Layout layout, int i) {
        int3 p = layout.blocks[i >> 9].coordinate;
        return {p.x * 8 + (i & 7), p.y * 8 + ((i >> 3) & 7), p.z * 8 + ((i >> 6) & 7)};
    }
    __device__ inline float3 center(Layout layout, int i) {
        int3 p  = cell_coordinate(layout, i);
        float h = spacing(layout, i);
        return {(p.x + .5f) * h, (p.y + .5f) * h, (p.z + .5f) * h};
    }
    __device__ inline int transverse(int local, int axis) {
        return axis == 0 ? ((local >> 3) & 7) + 8 * (local >> 6) : axis == 1 ? (local & 7) + 8 * (local >> 6) : (local & 7) + (local & 56);
    }
    __device__ inline int negative_face(int i, int axis) {
        return (i >> 9) * 1728 + axis * 576 + (i & 511);
    }
    __device__ inline int positive_face(int i, int axis) {
        return (i >> 9) * 1728 + axis * 576 + 512 + transverse(i & 511, axis);
    }
    __device__ inline Links cell_faces(Grid grid, int i, int axis, int sign) {
        int v = (i >> (3 * axis)) & 7;
        if ((sign < 0 && v > 0) || (sign > 0 && v < 7)) return {{negative_face(i + (sign > 0 ? 1 << (3 * axis) : 0), axis), -1, -1, -1}};
        return grid.links[(i >> 9) * 384 + (axis * 2 + (sign > 0)) * 64 + transverse(i & 511, axis)];
    }
    __device__ inline void face_position(Layout layout, int f, float3& p, float& h, int& axis, int& owner, int& sign) {
        int b = f / 1728, a = (f % 1728) / 576, local = f % 576;
        axis = a;
        sign = local < 512 ? -1 : 1;
        if (local >= 512) {
            int t = local - 512, u = t & 7, v = t >> 3;
            local = a == 0 ? 7 + u * 8 + v * 64 : a == 1 ? u + 56 + v * 64 : u + v * 8 + 448;
        }
        owner = b * 512 + local;
        h     = spacing(layout, owner);
        p     = center(layout, owner);
        if (axis == 0) p.x += sign * .5f * h;
        if (axis == 1) p.y += sign * .5f * h;
        if (axis == 2) p.z += sign * .5f * h;
    }
    __device__ inline float3 axis_vector(int axis, float amount) {
        return {axis == 0 ? amount : 0, axis == 1 ? amount : 0, axis == 2 ? amount : 0};
    }
    __device__ inline float sample_boundary(Grid grid, float3 p, float time) {
        int i = locate(grid.layout, p);
        if (i < 0) return 4 * grid.layout.h;
        int block   = i >> 9;
        int3 origin = grid.layout.blocks[block].coordinate;
        float3 q    = p / spacing(grid.layout, i) - float3{float(origin.x * 8), float(origin.y * 8), float(origin.z * 8)};
        int3 base{min(7, max(0, int(floorf(q.x)))), min(7, max(0, int(floorf(q.y)))), min(7, max(0, int(floorf(q.z))))};
        float3 t{q.x - base.x, q.y - base.y, q.z - base.z};
        float fraction = grid.staticGeometry ? 0 : fminf(2, fmaxf(0, 2 * (time - grid.geometryTime) / grid.geometryDuration));
        int lower      = fraction < 1 ? 0 : 1;
        float blend = fraction - lower, sum = 0;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    int index   = block * 729 + base.x + x + 9 * (base.y + y) + 81 * (base.z + z);
                    float value = grid.distance[lower][index];
                    if (!grid.staticGeometry) value += (grid.distance[lower + 1][index] - value) * blend;
                    sum += value * (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y) * (z ? t.z : 1 - t.z);
                }
        return sum;
    }
    __device__ inline float side_velocity(Grid grid, Fields fields, int i, int axis, int sign, bool total = false) {
        Links links = cell_faces(grid, i, axis, sign);
        float sum   = 0;
        for (int f : links.face)
            if (f >= 0) {
                float faceH = ldexpf(grid.layout.h, grid.layout.blocks[f / 1728].level);
                sum += total ? grid.faces[f].area * fields.velocity[f] + grid.faces[f].solidFlux : fields.velocity[f] * faceH * faceH;
            }
        float h = spacing(grid.layout, i);
        return sum / (h * h);
    }
    __device__ inline float minmod(float a, float b) {
        return a * b > 0 ? copysignf(fminf(fabsf(a), fabsf(b)), a) : 0;
    }
    __device__ inline float3 scalar_gradient(Grid grid, const float* field, int i) {
        if (i < 0 || field[i] == 0) return {};
        float q = field[i], h = spacing(grid.layout, i);
        float3 c = center(grid.layout, i);
        float slopes[3], extent = 0;
        for (int a = 0; a < 3; ++a) {
            int l = locate(grid.layout, c - axis_vector(a, h)), r = locate(grid.layout, c + axis_vector(a, h));
            float dl  = l >= 0 ? (q - field[l]) / (component(c - center(grid.layout, l), a)) : 0;
            float dr  = r >= 0 ? (field[r] - q) / (component(center(grid.layout, r) - c, a)) : 0;
            slopes[a] = minmod(dl, dr);
            extent += fabsf(slopes[a]) * h * .5f;
        }
        float limiter = extent > q && q >= 0 ? q / extent : 1;
        return {limiter * slopes[0], limiter * slopes[1], limiter * slopes[2]};
    }
    __device__ inline float face_value(Grid grid, Fields fields, float3 p, int axis, float background) {
        float epsilon = grid.layout.h * .001f;
        int i = locate(grid.layout, p + axis_vector(axis, epsilon)), left = locate(grid.layout, p - axis_vector(axis, epsilon));
        if (left >= 0 && (i < 0 || spacing(grid.layout, left) < spacing(grid.layout, i))) i = left;
        if (i < 0) return background;
        float t   = .5f + component(p - center(grid.layout, i), axis) / spacing(grid.layout, i);
        float low = side_velocity(grid, fields, i, axis, -1), high = side_velocity(grid, fields, i, axis, 1);
        return low + (high - low) * t;
    }
    __device__ inline float native_sample(Grid grid, Fields fields, float3 p, int field, int anchor, float* lower = nullptr, float* upper = nullptr, bool* uniform = nullptr) {
        int block  = anchor >> 9;
        Block node = grid.layout.blocks[block];
        float h    = spacing(grid.layout, anchor);
        float3 stagger{.5f, .5f, .5f};
        if (field < 3) stagger = stagger - axis_vector(field, .5f);
        float3 q = p / h - stagger;
        int3 base{int(floorf(q.x)), int(floorf(q.y)), int(floorf(q.z))};
        float3 t{q.x - base.x, q.y - base.y, q.z - base.z};
        int first = (block * 5 + field) * 1331 + base.x - node.coordinate.x * 8 + 1 + 11 * (base.y - node.coordinate.y * 8 + 1) + 121 * (base.z - node.coordinate.z * 8 + 1);
        float sum = 0, lo = 1.e30f, hi = -1.e30f;
        bool sameLevel = true;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    int index   = first + x + 11 * y + 121 * z;
                    float value = fields.samples[index];
                    sameLevel &= grid.uniformSamples[index] != 0;
                    sum += value * (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y) * (z ? t.z : 1 - t.z);
                    lo = fminf(lo, value);
                    hi = fmaxf(hi, value);
                }
        if (lower) *lower = lo;
        if (upper) *upper = hi;
        if (uniform) *uniform = sameLevel;
        return sum;
    }
    // The common h/2 lattice contains EVERY native cell/face center. Ordinary
    // native trilinear interpolation is exact away from transitions. At a level
    // transition its nodal extension provides a continuous interpolant without
    // storing a dense grid or smoothing the identity map at dt=0.
    __device__ inline float sample(Grid grid, Fields fields, float3 p, int field, float background = 0, float* lower = nullptr, float* upper = nullptr) {
        int anchor = locate(grid.layout, p);
        if (anchor >= 0) {
            bool uniform;
            float value = native_sample(grid, fields, p, field, anchor, lower, upper, &uniform);
            if (uniform) return value;
        }
        float h  = .5f * grid.layout.h;
        float3 q = p / h;
        int3 base{int(floorf(q.x)), int(floorf(q.y)), int(floorf(q.z))};
        float3 t{q.x - base.x, q.y - base.y, q.z - base.z};
        float sum = 0, lo = 1.e30f, hi = -1.e30f;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    float3 point{(base.x + x) * h, (base.y + y) * h, (base.z + z) * h};
                    int i = locate(grid.layout, point);
                    if (field < 3) {
                        int left = locate(grid.layout, point - axis_vector(field, grid.layout.h * .001f));
                        if (left >= 0 && (i < 0 || spacing(grid.layout, left) < spacing(grid.layout, i))) i = left;
                    }
                    if (i < 0)
                        for (int corner = 1; corner < 8 && i < 0; ++corner) i = locate(grid.layout, point - float3{(corner & 1 ? .002f : 0) * h, (corner & 2 ? .002f : 0) * h, (corner & 4 ? .002f : 0) * h});
                    float value = i < 0 ? background : native_sample(grid, fields, point, field, i);
                    sum += value * (x ? t.x : 1 - t.x) * (y ? t.y : 1 - t.y) * (z ? t.z : 1 - t.z);
                    lo = fminf(lo, value);
                    hi = fmaxf(hi, value);
                }
        if (lower) *lower = lo;
        if (upper) *upper = hi;
        return sum;
    }
    __device__ inline float3 sample_velocity(Grid grid, Fields fields, float3 p, float3 wind) {
        return {sample(grid, fields, p, 0, wind.x), sample(grid, fields, p, 1, wind.y), sample(grid, fields, p, 2, wind.z)};
    }
#endif
} // namespace pyro::cuda::sparse
#endif
