#include "cuda.cuh"

namespace pyro::cuda {
    __global__ void initialize_pool(int* freeBlocks, int* freeCount, int* age, int capacity) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < capacity) {
            freeBlocks[i] = capacity - 1 - i;
            age[i]        = 0;
        }
        if (i == 0) *freeCount = capacity;
    }

    __global__ void neighbor_kernel(Layout layout) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= layout.blocks * 6) return;
        int id = layout.active[i / 6], direction = i % 6;
        layout.neighbors[id * 6 + direction] = find_block(layout, offset(layout.coordinates[id], direction / 2, direction % 2 ? 1 : -1));
    }

    __device__ void request_block(Grid grid, int3 p, unsigned role, int* error) {
        auto wanted = key(p);
        int slot    = hash_slot(wanted, grid.layout.tableCapacity);
        for (int probe = 0; probe < grid.layout.tableCapacity; ++probe) {
            auto previous = atomicCAS(grid.nextKeys + slot, emptyKey, wanted);
            if (previous == emptyKey || previous == wanted) {
                atomicOr(grid.requested + slot, role);
                return;
            }
            slot = (slot + 1) & (grid.layout.tableCapacity - 1);
        }
        atomicExch(error, 1);
    }

    __device__ void request_domain_block(Grid grid, int3 p, unsigned reason, int* error) {
        // Positive storage neighbors own outer MAC faces and interpolation corners.
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) request_block(grid, {p.x + x, p.y + y, p.z + z}, x || y || z ? 1u : 3u | reason, error);
    }

    __global__ void request_existing(Grid grid, Config config, float dt, int* error) {
        int id = grid.layout.active[blockIdx.x];
        __shared__ float maxima[3][threads];
        float smoke = 0.0f, heat = 0.0f, speed = 0.0f;
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            smoke = fmaxf(smoke, grid.state.smoke[i]);
            heat  = fmaxf(heat, fabsf(grid.state.temperature[i]));
            float3 u{grid.state.velocity[0][i] - config.wind.x, grid.state.velocity[1][i] - config.wind.y, grid.state.velocity[2][i] - config.wind.z};
            if (grid.fluid[i]) speed = fmaxf(speed, length(u));
        }
        maxima[0][threadIdx.x] = smoke;
        maxima[1][threadIdx.x] = heat;
        maxima[2][threadIdx.x] = speed;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                for (int j = 0; j < 3; ++j) maxima[j][threadIdx.x] = fmaxf(maxima[j][threadIdx.x], maxima[j][threadIdx.x + stride]);
            __syncthreads();
        }
        __shared__ int radius, retained;
        __shared__ unsigned reason;
        if (threadIdx.x == 0) {
            reason      = (maxima[0][0] > config.smokeThreshold ? gridSmoke : 0u) | (maxima[1][0] > config.temperatureThreshold ? gridTemperature : 0u) | (maxima[2][0] > config.velocityThreshold ? gridVelocity : 0u);
            bool active = reason != 0;
            // Only former core blocks receive hysteresis, not every allocated halo.
            if (active) grid.age[id] = config.keepAliveSteps + 1;
            else if (grid.age[id] > 0) --grid.age[id];
            retained = active || grid.age[id] > 0;
            if (!active && retained) reason = gridRetained;
            radius = max(config.pressurePadding, static_cast<int>(ceilf(2.0f * (maxima[2][0] + length(config.wind)) * dt / (8.0f * config.cellSize))) + 1);
        }
        __syncthreads();
        if (!retained) return;
        int width   = 2 * radius + 1;
        int3 origin = grid.layout.coordinates[id];
        for (int i = threadIdx.x; i < width * width * width; i += blockDim.x) {
            int3 p{origin.x + i % width - radius, origin.y + (i / width) % width - radius, origin.z + i / (width * width) - radius};
            request_domain_block(grid, p, p.x == origin.x && p.y == origin.y && p.z == origin.z ? reason : gridPadding, error);
        }
    }

    __global__ void request_inputs(Grid grid, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, float dt, int* error) {
        int item = blockIdx.x;
        float3 lo{}, hi{};
        if (item < sourceCount) {
            Source source = sources[item];
            if (time + dt < source.startTime || time >= source.endTime) return;
            float3 e = source.shape == Shape::sphere ? float3{source.halfExtent.x, source.halfExtent.x, source.halfExtent.x} : source.halfExtent;
            lo       = source.center - e;
            hi       = source.center + e;
            if (source.shape == Shape::mesh) {
                lo = source.motion.swept.lower;
                hi = source.motion.swept.upper;
            }
        } else {
            Collider collider = colliders[item - sourceCount];
            float radius      = collider.shape == Shape::sphere ? collider.halfExtent.x : length(collider.halfExtent);
            float3 a          = collider.center + collider.linearVelocity * (time - collider.poseTime);
            float3 b          = a + collider.linearVelocity * dt;
            lo                = {fminf(a.x, b.x) - radius, fminf(a.y, b.y) - radius, fminf(a.z, b.z) - radius};
            hi                = {fmaxf(a.x, b.x) + radius, fmaxf(a.y, b.y) + radius, fmaxf(a.z, b.z) + radius};
            if (collider.shape == Shape::mesh) {
                lo = collider.motion.swept.lower;
                hi = collider.motion.swept.upper;
            }
        }
        float spacing = config.cellSize * 8.0f;
        int padding   = max(config.pressurePadding, 1);
        int3 a{static_cast<int>(floorf(lo.x / spacing)) - padding, static_cast<int>(floorf(lo.y / spacing)) - padding, static_cast<int>(floorf(lo.z / spacing)) - padding};
        int3 b{static_cast<int>(floorf(hi.x / spacing)) + padding, static_cast<int>(floorf(hi.y / spacing)) + padding, static_cast<int>(floorf(hi.z / spacing)) + padding};
        int3 n{b.x - a.x + 1, b.y - a.y + 1, b.z - a.z + 1};
        for (int i = threadIdx.x; i < n.x * n.y * n.z; i += blockDim.x) {
            int3 p{a.x + i % n.x, a.y + (i / n.x) % n.y, a.z + i / (n.x * n.y)};
            bool interior = p.x >= a.x + padding && p.x <= b.x - padding && p.y >= a.y + padding && p.y <= b.y - padding && p.z >= a.z + padding && p.z <= b.z - padding;
            request_domain_block(grid, p, interior ? (item < sourceCount ? gridEmitter : gridCollider) : gridPadding, error);
        }
    }

    __global__ void retain_blocks(Grid grid, Layout next) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.blocks) return;
        int id   = grid.layout.active[i];
        int slot = find_slot(next, grid.layout.coordinates[id]);
        if (slot >= 0) next.indices[slot] = id;
        else grid.freeBlocks[atomicAdd(grid.freeCount, 1)] = id;
    }

    __global__ void assign_blocks(Grid grid, Layout next, int* error) {
        int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (slot >= next.tableCapacity || next.keys[slot] == emptyKey) return;
        int id = next.indices[slot];
        if (id < 0) {
            int freeIndex = atomicSub(grid.freeCount, 1) - 1;
            if (freeIndex < 0) {
                atomicExch(error, 1);
                return;
            }
            id                     = grid.freeBlocks[freeIndex];
            next.indices[slot]     = id;
            next.coordinates[id]   = coordinate(next.keys[slot]);
            grid.age[id]           = 0;
            grid.roles[id]         = grid.requested[slot] | 4u; // Newly allocated.
            grid.boundaryDirty[id] = 1;
        } else {
            grid.boundaryDirty[id] = ((grid.roles[id] ^ grid.requested[slot]) & 2u) != 0;
            grid.roles[id]         = grid.requested[slot];
        }
        int position = atomicAdd(next.count, 1);
        if (position < next.capacity) next.active[position] = id;
        else atomicExch(error, 1);
    }

    __global__ void initialize_blocks(Grid grid, float3 wind) {
        int id = grid.layout.active[blockIdx.x];
        if (!threadIdx.x && !(grid.roles[id] & 4u) && !grid.boundaryDirty[id]) {
            // Corner SDF reads can cross positive edges and diagonals, not just faces.
            Layout previous  = grid.layout;
            previous.keys    = grid.nextKeys;
            previous.indices = grid.nextIndices;
            int3 c           = grid.layout.coordinates[id];
            for (int k = 1; k < 8; ++k) {
                int3 p{c.x + (k & 1), c.y + ((k >> 1) & 1), c.z + ((k >> 2) & 1)};
                if ((find_block(previous, p) >= 0) != (find_block(grid.layout, p) >= 0)) grid.boundaryDirty[id] = 1;
            }
        }
        if (!(grid.roles[id] & 4u)) return;
        for (int j = threadIdx.x; j < blockCells; j += blockDim.x) {
            int i         = id * blockCells + j;
            grid.fluid[i] = 0;
            Fields buffers[]{grid.state, grid.midpoint, grid.next, grid.transported};
            for (auto fields : buffers) {
                fields.smoke[i]       = 0.0f;
                fields.temperature[i] = 0.0f;
                for (int axis = 0; axis < 3; ++axis) fields.velocity[axis][i] = component(wind, axis);
            }
            for (int axis = 0; axis < 3; ++axis) {
                grid.aperture[axis][i]      = 1.0f;
                grid.solidVelocity[axis][i] = 0.0f;
            }
        }
    }

    __global__ void geometry_kernel(Grid grid, float h, const Collider* colliders, int count, float time, int sample, bool onlyNew) {
        int id = grid.layout.active[blockIdx.x];
        if (onlyNew && !(grid.roles[id] & 4u)) return;
        for (int j = threadIdx.x; j < blockCells; j += blockDim.x) {
            int index                    = id * blockCells + j;
            int3 cell                    = cell_coordinate(grid.layout, index);
            grid.distance[sample][index] = fminf(4.0f * h, fmaxf(-4.0f * h, solid_distance({cell.x * h, cell.y * h, cell.z * h}, time, colliders, count)));
        }
    }

    __global__ void boundary_kernel(Grid grid, float h, const Collider* colliders, int count, float time, int sample, bool onlyDirty) {
        int id = grid.layout.active[blockIdx.x];
        if (onlyDirty && !grid.boundaryDirty[id]) return;
        for (int j = threadIdx.x; j < blockCells; j += blockDim.x) {
            int index = id * blockCells + j;
            int3 cell = cell_coordinate(grid.layout, index);
            float phi[8], maximum = -1.0e20f;
            for (int k = 0; k < 8; ++k) {
                phi[k]  = value(grid.distance[sample], cell_index(grid.layout, {cell.x + (k & 1), cell.y + ((k >> 1) & 1), cell.z + ((k >> 2) & 1)}), 4.0f * h);
                maximum = fmaxf(maximum, phi[k]);
            }
            grid.fluid[index] = (grid.roles[id] & 2u) && maximum > 0.0f;
            float volume      = (tetrahedron_fraction(phi[0], phi[1], phi[3], phi[7]) + tetrahedron_fraction(phi[0], phi[3], phi[2], phi[7]) + tetrahedron_fraction(phi[0], phi[2], phi[6], phi[7]) + tetrahedron_fraction(phi[0], phi[6], phi[4], phi[7]) + tetrahedron_fraction(phi[0], phi[4], phi[5], phi[7]) + tetrahedron_fraction(phi[0], phi[5], phi[1], phi[7])) / 6.0f;
            float divergence  = 0;
            if (volume < 1.0f) solid_distance({(cell.x + 0.5f) * h, (cell.y + 0.5f) * h, (cell.z + 0.5f) * h}, time, colliders, count, nullptr, &divergence);
            grid.solidDivergence[index] = (1.0f - volume) * divergence;
            for (int axis = 0; axis < 3; ++axis) {
                int b = 1 << ((axis + 1) % 3), c = 1 << ((axis + 2) % 3);
                grid.aperture[axis][index] = 0.5f * (triangle_fraction(phi[0], phi[b], phi[b | c]) + triangle_fraction(phi[0], phi[b | c], phi[c]));
                float3 p{cell.x + 0.5f, cell.y + 0.5f, cell.z + 0.5f};
                if (axis == 0) p.x -= 0.5f;
                if (axis == 1) p.y -= 0.5f;
                if (axis == 2) p.z -= 0.5f;
                float covered = 1.0f - grid.aperture[axis][index];
                if (covered > 0.0f && covered < 1.0f) {
                    float3 y{float(b & 1), float((b >> 1) & 1), float((b >> 2) & 1)};
                    float3 z{float(c & 1), float((c >> 1) & 1), float((c >> 2) & 1)};
                    float3 moment = (solid_face_moment(phi[0], phi[b], phi[b | c], {}, y, y + z) + solid_face_moment(phi[0], phi[b | c], phi[c], {}, y + z, z)) * 0.5f;
                    p             = float3{float(cell.x), float(cell.y), float(cell.z)} + moment / covered;
                }
                float3 velocity{};
                if (covered > 0.0f) solid_distance(p * h, time, colliders, count, &velocity);
                grid.solidVelocity[axis][index] = component(velocity, axis);
            }
        }
        __syncthreads();
        if (!threadIdx.x) grid.boundaryDirty[id] = 0;
    }

    __global__ void density_kernel(Grid grid, DensityBlock* destination, int* count) {
        int id        = grid.layout.active[blockIdx.x];
        bool occupied = false;
        for (int j = threadIdx.x; j < blockCells; j += blockDim.x) occupied |= grid.state.smoke[id * blockCells + j] != 0.0f;
        if (!__syncthreads_or(occupied)) return;
        __shared__ int output;
        if (threadIdx.x == 0) {
            output                         = atomicAdd(count, 1);
            destination[output].coordinate = grid.layout.coordinates[id];
        }
        __syncthreads();
        for (int j = threadIdx.x; j < blockCells; j += blockDim.x) destination[output].values[j] = grid.state.smoke[id * blockCells + j];
    }

    void build_neighbors(Layout layout, cudaStream_t stream) {
        if (layout.blocks) neighbor_kernel<<<(layout.blocks * 6 + threads - 1) / threads, threads, 0, stream>>>(layout);
    }

    void request_topology(Grid grid, Config config, const Source* sources, int sourceCount, const Collider* colliders, int colliderCount, float time, float dt, int* error, cudaStream_t stream) {
        if (grid.layout.blocks == 0) initialize_pool<<<(grid.layout.capacity + threads - 1) / threads, threads, 0, stream>>>(grid.freeBlocks, grid.freeCount, grid.age, grid.layout.capacity);
        cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
        cuda_check(cudaMemsetAsync(grid.nextKeys, 255, grid.layout.tableCapacity * sizeof(unsigned long long), stream));
        cuda_check(cudaMemsetAsync(grid.nextIndices, 255, grid.layout.tableCapacity * sizeof(int), stream));
        cuda_check(cudaMemsetAsync(grid.requested, 0, grid.layout.tableCapacity * sizeof(unsigned), stream));
        if (grid.layout.blocks) request_existing<<<grid.layout.blocks, threads, 0, stream>>>(grid, config, dt, error);
        if (sourceCount + colliderCount) request_inputs<<<sourceCount + colliderCount, threads, 0, stream>>>(grid, config, sources, sourceCount, colliders, colliderCount, time, dt, error);
        Layout next  = grid.layout;
        next.keys    = grid.nextKeys;
        next.indices = grid.nextIndices;
        if (grid.layout.blocks) retain_blocks<<<(grid.layout.blocks + threads - 1) / threads, threads, 0, stream>>>(grid, next);
        cuda_check(cudaMemsetAsync(next.count, 0, sizeof(int), stream));
        assign_blocks<<<(next.tableCapacity + threads - 1) / threads, threads, 0, stream>>>(grid, next, error);
    }

    void initialize_topology(Grid grid, float3 wind, cudaStream_t stream) {
        if (grid.layout.blocks) initialize_blocks<<<grid.layout.blocks, threads, 0, stream>>>(grid, wind);
        build_neighbors(grid.layout, stream);
    }

    void sample_geometry(Grid grid, float h, const Collider* colliders, int count, float time, int sample, bool onlyNew, cudaStream_t stream) {
        if (grid.layout.blocks) geometry_kernel<<<grid.layout.blocks, threads, 0, stream>>>(grid, h, colliders, count, time, sample, onlyNew);
    }

    void update_boundary(Grid grid, float h, const Collider* colliders, int count, float time, int sample, bool onlyDirty, cudaStream_t stream) {
        if (grid.layout.blocks) boundary_kernel<<<grid.layout.blocks, threads, 0, stream>>>(grid, h, colliders, count, time, sample, onlyDirty);
    }

    __global__ void state_kernel(Grid grid, StateBlock* blocks, bool restore) {
        int ordinal = blockIdx.x;
        int id      = restore ? blocks[ordinal].id : grid.layout.active[ordinal];
        if (!threadIdx.x) {
            if (restore) {
                grid.layout.active[ordinal] = id;
                grid.layout.coordinates[id] = blocks[ordinal].coordinate;
                grid.roles[id]              = blocks[ordinal].role;
                grid.age[id]                = blocks[ordinal].age;
            } else {
                blocks[ordinal].id         = id;
                blocks[ordinal].coordinate = grid.layout.coordinates[id];
                blocks[ordinal].role       = grid.roles[id];
                blocks[ordinal].age        = grid.age[id];
            }
        }
        for (int j = threadIdx.x; j < blockCells; j += blockDim.x) {
            int i = id * blockCells + j;
            if (restore) {
                grid.state.smoke[i]       = blocks[ordinal].smoke[j];
                grid.state.temperature[i] = blocks[ordinal].temperature[j];
                for (int axis = 0; axis < 3; ++axis) grid.state.velocity[axis][i] = blocks[ordinal].velocity[axis][j];
            } else {
                blocks[ordinal].smoke[j]       = grid.state.smoke[i];
                blocks[ordinal].temperature[j] = grid.state.temperature[i];
                for (int axis = 0; axis < 3; ++axis) blocks[ordinal].velocity[axis][j] = grid.state.velocity[axis][i];
            }
        }
    }

    __global__ void grid_kernel(Grid grid, GridBlock* blocks) {
        int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
        if (ordinal >= grid.layout.blocks) return;
        int id          = grid.layout.active[ordinal];
        blocks[ordinal] = {grid.layout.coordinates[id], grid.roles[id] & ~4u, max(grid.age[id] - 1, 0)};
    }

    void pack_density(Grid grid, DensityBlock* destination, int* count, cudaStream_t stream) {
        cuda_check(cudaMemsetAsync(count, 0, sizeof(int), stream));
        if (grid.layout.blocks) density_kernel<<<grid.layout.blocks, threads, 0, stream>>>(grid, destination, count);
    }
    void transfer_state(Grid grid, StateBlock* blocks, bool restore, cudaStream_t stream) {
        if (grid.layout.blocks) state_kernel<<<grid.layout.blocks, threads, 0, stream>>>(grid, blocks, restore);
    }
    void pack_grid(Grid grid, GridBlock* blocks, cudaStream_t stream) {
        if (grid.layout.blocks) grid_kernel<<<(grid.layout.blocks + threads - 1) / threads, threads, 0, stream>>>(grid, blocks);
    }
} // namespace pyro::cuda
