#include "cuda.cuh"
#include <algorithm>
#include <cub/cub.cuh>

namespace pyro::cuda::sparse {
    void initialize_scratch(Scratch& scratch, int capacity, cudaStream_t stream) {
        scratch.capacity = capacity;
        for (auto& p : scratch.keys)
            if (auto cuda_result = cudaMallocAsync(&p, capacity * sizeof(unsigned long long), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        for (auto& p : scratch.values)
            if (auto cuda_result = cudaMallocAsync(&p, capacity * sizeof(float), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaMallocAsync(&scratch.count, sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        std::size_t sort_bytes = 0, unique_bytes = 0;
        if (auto cuda_result = cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, scratch.keys[0], scratch.keys[1], scratch.values[0], scratch.values[1], capacity, 0, 64, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cub::DeviceSelect::Unique(nullptr, unique_bytes, scratch.keys[1], scratch.keys[2], scratch.count, capacity, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        scratch.sort_bytes = std::max(sort_bytes, unique_bytes);
        if (auto cuda_result = cudaMallocAsync(&scratch.sort_storage, scratch.sort_bytes, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
    }
    __global__ void hash_blocks(Layout layout) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= layout.count) return;
        auto block           = layout.blocks[b];
        int slot             = insert(layout.keys, layout.table_size, key(block.coordinate, block.level));
        layout.indices[slot] = b;
    }
    void rebuild_hash(Layout layout, cudaStream_t stream) {
        if (auto cuda_result = cudaMemsetAsync(layout.keys, 255, layout.table_size * sizeof(unsigned long long), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (layout.count) hash_blocks<<<(layout.count + 255) / 256, 256, 0, stream>>>(layout);
    }
    __device__ float restriction(const float* field, int b, int3 p) {
        float sum = 0;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) sum += field[b * 512 + 2 * p.x + x + 8 * (2 * p.y + y) + 64 * (2 * p.z + z)];
        return sum * .125f;
    }
    __global__ void classify(Grid grid, Topology topology, Config config) {
        int b      = blockIdx.x;
        auto block = grid.layout.blocks[b];
        __shared__ float errors[256];
        __shared__ int active[256];
        float error = 0;
        bool live   = false;
        for (int local = threadIdx.x; local < 512; local += blockDim.x) {
            int i = b * 512 + local;
            live |= fabsf(grid.state.smoke[i]) > config.smoke_threshold || fabsf(grid.state.temperature[i]) > config.temperature_threshold;
            int3 p{(local & 7) >> 1, ((local >> 3) & 7) >> 1, (local >> 6) >> 1};
            for (int f = 0; f < 2; ++f) {
                const float* values = f ? grid.state.temperature : grid.state.smoke;
                float average = restriction(values, b, p), estimate = average, scale = fmaxf(fabsf(average), fabsf(values[i]));
                for (int axis = 0; axis < 3; ++axis) {
                    int v       = axis == 0 ? p.x : axis == 1 ? p.y : p.z;
                    float l     = restriction(values, b, int3{p.x + (axis == 0 ? (v ? -1 : 0) : 0), p.y + (axis == 1 ? (v ? -1 : 0) : 0), p.z + (axis == 2 ? (v ? -1 : 0) : 0)});
                    float r     = restriction(values, b, int3{p.x + (axis == 0 ? (v < 3 ? 1 : 0) : 0), p.y + (axis == 1 ? (v < 3 ? 1 : 0) : 0), p.z + (axis == 2 ? (v < 3 ? 1 : 0) : 0)});
                    float slope = v == 0 ? r - average : v == 3 ? average - l : minmod(average - l, r - average);
                    estimate += (((local >> (3 * axis)) & 1) ? .25f : -.25f) * slope;
                    scale = fmaxf(scale, fmaxf(fabsf(l), fabsf(r)));
                }
                float tolerance = (f ? config.temperature_absolute_error : config.smoke_absolute_error) + config.relative_error * scale;
                error           = fmaxf(error, fabsf(values[i] - estimate) / tolerance);
            }
            for (int axis = 0; axis < 3; ++axis) {
                float l = side_velocity(grid, grid.state, i, axis, -1), r = side_velocity(grid, grid.state, i, axis, 1);
                float v = .5f * (l + r), wind = (axis == 0 ? config.wind.x : axis == 1 ? config.wind.y : config.wind.z);
                live |= fabsf(v - wind) > config.velocity_threshold;
                float h  = spacing(grid.layout, i);
                float3 c = center(grid.layout, i);
                for (int a = 0; a < 3; ++a) {
                    int j = locate(grid.layout, float3{c.x - (a == 0 ? h : 0), c.y - (a == 1 ? h : 0), c.z - (a == 2 ? h : 0)}), k = locate(grid.layout, float3{c.x + (a == 0 ? h : 0), c.y + (a == 1 ? h : 0), c.z + (a == 2 ? h : 0)});
                    float left  = j < 0 ? wind : .5f * (side_velocity(grid, grid.state, j, axis, -1) + side_velocity(grid, grid.state, j, axis, 1));
                    float right = k < 0 ? wind : .5f * (side_velocity(grid, grid.state, k, axis, -1) + side_velocity(grid, grid.state, k, axis, 1));
                    error       = fmaxf(error, .125f * fabsf(left - 2 * v + right) / (config.velocity_absolute_error + config.relative_error * fmaxf(fabsf(v), fmaxf(fabsf(left), fabsf(right)))));
                }
            }
        }
        errors[threadIdx.x] = error;
        active[threadIdx.x] = live;
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                errors[threadIdx.x] = fmaxf(errors[threadIdx.x], errors[threadIdx.x + stride]);
                active[threadIdx.x] |= active[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x) return;
        int quiet             = errors[0] < .5f ? block.quiet + 1 : 0;
        topology.quiet[b]     = quiet;
        topology.remaining[b] = active[0] ? config.keep_alive : max(0, block.remaining - 1);
        topology.decision[b]  = errors[0] > 1 && block.level > 0 ? block.level - 1 : quiet >= config.coarsen_delay ? min(block.level + 1, config.levels - 1) : block.level;
    }
    __device__ void root_box(Topology topology, Config config, float3 low, float3 high) {
        float h = ldexpf(config.cell_size * 8, config.levels - 1);
        int3 l{int(floorf(low.x / h)) - config.pressure_padding, int(floorf(low.y / h)) - config.pressure_padding, int(floorf(low.z / h)) - config.pressure_padding};
        int3 u{int(floorf(high.x / h)) + config.pressure_padding, int(floorf(high.y / h)) + config.pressure_padding, int(floorf(high.z / h)) + config.pressure_padding};
        for (int z = l.z; z <= u.z; ++z)
            for (int y = l.y; y <= u.y; ++y)
                for (int x = l.x; x <= u.x; ++x)
                    if (insert(topology.roots, topology.table_size, key({x, y, z}, config.levels - 1)) < 0) atomicExch(topology.counters + 1, 1);
    }
    __global__ void request_old(Grid grid, Topology topology, Config config) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= grid.layout.count) return;
        auto block  = grid.layout.blocks[b];
        int desired = topology.decision[b];
        for (int l = desired + 1; l < config.levels; ++l) {
            int shift = l - block.level;
            int3 p    = block.coordinate;
            if (shift >= 0) p = {p.x >> shift, p.y >> shift, p.z >> shift};
            if (insert(topology.refine, topology.table_size, key(p, l)) < 0) atomicExch(topology.counters + 1, 1);
        }
        if (!topology.remaining[b]) return;
        float h = ldexpf(config.cell_size * 8, block.level), margin = config.cell_size * (2 + config.cfl * config.adapt_interval);
        float3 low{block.coordinate.x * h - margin, block.coordinate.y * h - margin, block.coordinate.z * h - margin};
        float3 high{low.x + (h + 2 * margin), low.y + (h + 2 * margin), low.z + (h + 2 * margin)};
        root_box(topology, config, low, high);
    }
    __global__ void request_scene(Topology topology, Config config, const Source* sources, int source_count, const Collider* colliders, int collider_count, float time, float horizon) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float3 low{}, high{};
        if (i < source_count) {
            Source source = sources[i];
            if (time + horizon < source.start_time || time > source.end_time) return;
            if (source.shape == Shape::mesh) {
                low  = source.motion.swept.lower;
                high = source.motion.swept.upper;
            } else {
                low  = {source.center.x - source.half_extent.x, source.center.y - source.half_extent.y, source.center.z - source.half_extent.z};
                high = {source.center.x + source.half_extent.x, source.center.y + source.half_extent.y, source.center.z + source.half_extent.z};
            }
        } else if (i < source_count + collider_count) {
            Collider body = colliders[i - source_count];
            if (body.shape == Shape::mesh) {
                low  = body.motion.swept.lower;
                high = body.motion.swept.upper;
            } else {
                float radius   = sqrtf(body.half_extent.x * body.half_extent.x + body.half_extent.y * body.half_extent.y + body.half_extent.z * body.half_extent.z);
                float interval = time - body.pose_time;
                float3 center{body.center.x + body.linear_velocity.x * interval, body.center.y + body.linear_velocity.y * interval, body.center.z + body.linear_velocity.z * interval};
                float3 next{center.x + body.linear_velocity.x * horizon, center.y + body.linear_velocity.y * horizon, center.z + body.linear_velocity.z * horizon};
                low  = {fminf(center.x, next.x) - radius, fminf(center.y, next.y) - radius, fminf(center.z, next.z) - radius};
                high = {fmaxf(center.x, next.x) + radius, fmaxf(center.y, next.y) + radius, fmaxf(center.z, next.z) + radius};
            }
        } else return;
        root_box(topology, config, low, high);
    }
    __global__ void collect_roots(Topology topology, Block* blocks) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= topology.table_size || topology.roots[i] == empty_key) return;
        int index = atomicAdd(topology.counters, 1);
        if (index >= topology.capacity) {
            atomicExch(topology.counters + 1, 1);
            return;
        }
        Node n        = node(topology.roots[i]);
        blocks[index] = {n.coordinate, n.level};
    }
    __global__ void mark_refinement(Layout layout, Topology topology, Config config, const Source* sources, int source_count, const Collider* colliders, int collider_count, float time, float horizon, float collider_speed, bool enforce_requests) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= layout.count) return;
        auto block = layout.blocks[b];
        if (!block.level) return;
        float h = ldexpf(config.cell_size, block.level), width = h * 8;
        float3 center{(block.coordinate.x + .5f) * width, (block.coordinate.y + .5f) * width, (block.coordinate.z + .5f) * width};
        bool split = enforce_requests && lookup(topology.refine, topology.table_size, key(block.coordinate, block.level)) >= 0;
        if (enforce_requests) {
            for (int i = 0; i < source_count && !split; ++i) {
                Source s = sources[i];
                if (time + horizon < s.start_time || time > s.end_time) continue;
                float support = width * 2.f + h * 8.f + s.motion.max_speed * horizon * 2;
                s.mesh.band   = support * .5f;
                split |= source_weight(s, center, support, time + .5f * horizon) > 0;
            }
            if (collider_count) {
                float band = 7.f * h + collider_speed * horizon;
                float d    = solid_distance(center, time + .5f * horizon, colliders, collider_count, nullptr, nullptr, band);
                split |= fabsf(d) < band;
            }
        }
        if (split) atomicExch(topology.split + b, 1);
    }
    __global__ void balance(Layout layout, int* split) {
        int b      = blockIdx.x;
        auto block = layout.blocks[b];
        int d      = threadIdx.x;
        if (d >= 27 || d == 13) return;
        int3 direction{d % 3 - 1, (d / 3) % 3 - 1, d / 9 - 1};
        float h = ldexpf(layout.h, block.level), width = h * 8;
        float3 p{(block.coordinate.x + .5f) * width + direction.x * (4 * h + layout.h * .01f), (block.coordinate.y + .5f) * width + direction.y * (4 * h + layout.h * .01f), (block.coordinate.z + .5f) * width + direction.z * (4 * h + layout.h * .01f)};
        int j = locate(layout, p);
        if (j >= 0 && layout.blocks[j >> 9].level > block.level + 1) atomicExch(split + (j >> 9), 1);
    }
    __global__ void split_blocks(Layout layout, Topology topology, Block* output) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= layout.count) return;
        auto block = layout.blocks[b];
        int n      = topology.split[b] ? 8 : 1;
        int begin  = atomicAdd(topology.counters, n);
        if (begin + n > topology.capacity) {
            atomicExch(topology.counters + 1, 1);
            return;
        }
        if (n == 1) output[begin] = block;
        else
            for (int child = 0; child < 8; ++child) output[begin + child] = {{block.coordinate.x * 2 + (child & 1), block.coordinate.y * 2 + ((child >> 1) & 1), block.coordinate.z * 2 + (child >> 2)}, block.level - 1};
    }
    __global__ void block_keys(Layout layout, unsigned long long* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < layout.count) keys[i] = key(layout.blocks[i].coordinate, layout.blocks[i].level);
    }
    __global__ void ordered_blocks(Layout old_layout, Layout layout, Topology topology, const unsigned long long* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= layout.count) return;
        Node n           = node(keys[i]);
        int old          = find_block(old_layout, n.coordinate, n.level);
        layout.blocks[i] = {n.coordinate, n.level, old >= 0 ? topology.quiet[old] : 0, old >= 0 ? topology.remaining[old] : 0};
        if (i >= old_layout.count || key(old_layout.blocks[i].coordinate, old_layout.blocks[i].level) != keys[i]) atomicExch(topology.counters + 2, 1);
    }
    int adapt(Grid old_grid, Grid new_grid, Topology topology, Scratch scratch, Config config, const Source* sources, int source_count, const Collider* colliders, int collider_count, float time, float horizon, bool& changed, cudaStream_t stream) {
        if (auto cuda_result = cudaMemsetAsync(topology.roots, 255, topology.table_size * sizeof(unsigned long long), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaMemsetAsync(topology.refine, 255, topology.table_size * sizeof(unsigned long long), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaMemsetAsync(topology.counters, 0, 2 * sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (old_grid.layout.count) {
            classify<<<old_grid.layout.count, 256, 0, stream>>>(old_grid, topology, config);
            request_old<<<(old_grid.layout.count + 255) / 256, 256, 0, stream>>>(old_grid, topology, config);
        }
        if (source_count + collider_count) request_scene<<<(source_count + collider_count + 255) / 256, 256, 0, stream>>>(topology, config, sources, source_count, colliders, collider_count, time, horizon);
        collect_roots<<<(topology.table_size + 255) / 256, 256, 0, stream>>>(topology, new_grid.layout.blocks);
        int status[2];
        if (auto cuda_result = cudaMemcpyAsync(status, topology.counters, sizeof(status), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (status[1]) return std::max(status[0], topology.capacity * 2);
        Layout layout = new_grid.layout;
        layout.count  = status[0];
        Block* output = topology.candidates;
        for (int pass = 0; pass < config.levels * 2 && layout.count; ++pass) {
            rebuild_hash(layout, stream);
            if (auto cuda_result = cudaMemsetAsync(topology.split, 0, layout.count * sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            mark_refinement<<<(layout.count + 255) / 256, 256, 0, stream>>>(layout, topology, config, sources, source_count, colliders, collider_count, time, horizon, old_grid.collider_speed, pass < config.levels);
            balance<<<layout.count, 32, 0, stream>>>(layout, topology.split);
            if (auto cuda_result = cudaMemsetAsync(topology.counters, 0, 2 * sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            split_blocks<<<(layout.count + 255) / 256, 256, 0, stream>>>(layout, topology, output);
            if (auto cuda_result = cudaMemcpyAsync(status, topology.counters, sizeof(status), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (status[1]) return status[0];
            std::swap(layout.blocks, output);
            bool done    = layout.count == status[0];
            layout.count = status[0];
            if (done) break;
        }
        if (auto cuda_result = cudaMemsetAsync(topology.counters + 2, 0, sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (layout.count) {
            block_keys<<<(layout.count + 255) / 256, 256, 0, stream>>>(layout, scratch.keys[0]);
            std::size_t bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceRadixSort::SortKeys(scratch.sort_storage, bytes, scratch.keys[0], scratch.keys[1], layout.count, 0, 64, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            new_grid.layout.count = layout.count;
            ordered_blocks<<<(layout.count + 255) / 256, 256, 0, stream>>>(old_grid.layout, new_grid.layout, topology, scratch.keys[1]);
        }
        new_grid.layout.count = layout.count;
        rebuild_hash(new_grid.layout, stream);
        int different;
        if (auto cuda_result = cudaMemcpyAsync(&different, topology.counters + 2, sizeof(int), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        changed = different || layout.count != old_grid.layout.count;
        return layout.count;
    }
    __global__ void connect_faces(Grid grid) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        float epsilon = sign * grid.layout.h * .01f;
        int other     = locate(grid.layout, float3{p.x + (axis == 0 ? epsilon : 0), p.y + (axis == 1 ? epsilon : 0), p.z + (axis == 2 ? epsilon : 0)});
        bool valid    = sign < 0 ? other < 0 || spacing(grid.layout, other) >= h : other < 0 || spacing(grid.layout, other) > h;
        Face face;
        if (valid) {
            face.left     = sign < 0 ? other : owner;
            face.right    = sign < 0 ? owner : other;
            face.distance = other < 0 ? .5f * h : .5f * (h + spacing(grid.layout, other));
            face.area     = h * h;
        }
        grid.faces[f] = face;
    }
    __global__ void connect_links(Grid grid) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= grid.layout.count * 384) return;
        int b = index / 384, direction = (index % 384) / 64, t = index & 63;
        int axis = direction / 2, sign = direction & 1 ? 1 : -1, v = sign < 0 ? 0 : 7;
        int local   = axis == 0 ? v + (t & 7) * 8 + (t >> 3) * 64 : axis == 1 ? (t & 7) + v * 8 + (t >> 3) * 64 : t + v * 64;
        int i       = b * 512 + local;
        float h     = spacing(grid.layout, i);
        float3 c    = center(grid.layout, i);
        float shift = sign * h * .5f, epsilon = sign * grid.layout.h * .01f;
        float3 p{c.x + (axis == 0 ? shift : 0), c.y + (axis == 1 ? shift : 0), c.z + (axis == 2 ? shift : 0)};
        int j = locate(grid.layout, float3{p.x + (axis == 0 ? epsilon : 0), p.y + (axis == 1 ? epsilon : 0), p.z + (axis == 2 ? epsilon : 0)});
        Links links;
        if (j < 0 || spacing(grid.layout, j) >= h) {
            links.face[0] = sign < 0 ? negative_face(i, axis) : j >= 0 && spacing(grid.layout, j) == h ? negative_face(j, axis) : positive_face(i, axis);
        } else
            for (int q = 0; q < 4; ++q) {
                int a = (axis + 1) % 3, b = (axis + 2) % 3;
                float u = (q & 1 ? .25f : -.25f) * h, v = (q & 2 ? .25f : -.25f) * h;
                float3 point{p.x + (axis == 0 ? epsilon : 0) + (a == 0 ? u : 0) + (b == 0 ? v : 0), p.y + (axis == 1 ? epsilon : 0) + (a == 1 ? u : 0) + (b == 1 ? v : 0), p.z + (axis == 2 ? epsilon : 0) + (a == 2 ? u : 0) + (b == 2 ? v : 0)};
                int child     = locate(grid.layout, point);
                links.face[q] = sign < 0 ? positive_face(child, axis) : negative_face(child, axis);
            }
        grid.links[index] = links;
        if (links.face[3] >= 0)
            for (int f : links.face) grid.faces[f].group = index;
    }
    void build_connectivity(Grid grid, cudaStream_t stream) {
        if (!grid.layout.count) return;
        connect_faces<<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid);
        connect_links<<<(grid.layout.count * 384 + 255) / 256, 256, 0, stream>>>(grid);
    }
    __global__ void sample_corners(Grid grid, Layout previous, const float* previous_distance, const Collider* colliders, int count, float time, int sample) {
        int b       = blockIdx.x;
        Block block = grid.layout.blocks[b];
        int old     = previous_distance ? find_block(previous, block.coordinate, block.level) : -1;
        float h = ldexpf(grid.layout.h, block.level), band = 4 * h;
        for (int local = threadIdx.x; local < 729; local += blockDim.x) {
            float value;
            if (old >= 0) value = previous_distance[old * 729 + local];
            else {
                float3 p{(block.coordinate.x * 8 + local % 9) * h, (block.coordinate.y * 8 + (local / 9) % 9) * h, (block.coordinate.z * 8 + local / 81) * h};
                value = solid_distance(p, time, colliders, count, nullptr, nullptr, band);
            }
            grid.distance[sample][b * 729 + local] = value;
        }
    }
    void sample_geometry(Grid grid, Layout previous, const float* previous_distance, const Collider* colliders, int count, float time, int sample, cudaStream_t stream) {
        if (grid.layout.count) sample_corners<<<grid.layout.count, 256, 0, stream>>>(grid, previous, previous_distance, colliders, count, time, sample);
    }
    __global__ void cell_geometry(Grid grid, const Collider* colliders, int count, float time, int sample) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        float h  = spacing(grid.layout, i);
        float3 p = center(grid.layout, i);
        int node = (i >> 9) * 729 + (i & 7) + 9 * ((i >> 3) & 7) + 81 * ((i >> 6) & 7);
        float d[8], maximum = -1.e30f;
        for (int k = 0; k < 8; ++k) {
            d[k]    = grid.distance[sample][node + (k & 1) + 9 * ((k >> 1) & 1) + 81 * (k >> 2)];
            maximum = fmaxf(maximum, d[k]);
        }
        constexpr int tetrahedra[6][4]{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}};
        float volume = 0;
        for (auto& t : tetrahedra) volume += tetrahedron_fraction(d[t[0]], d[t[1]], d[t[2]], d[t[3]]) / 6.f;
        float rate = 0;
        if (!grid.static_geometry && volume < 1 && maximum > 0) solid_distance(p, time, colliders, count, nullptr, &rate, 4 * h);
        grid.fluid[i]            = maximum > 0;
        grid.solid_divergence[i] = (1 - volume) * rate;
    }
    __global__ void face_geometry(Grid grid, const Collider* colliders, int count, float time, int sample) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.left < 0 && face.right < 0) return;
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        int3 local{owner & 7, (owner >> 3) & 7, (owner >> 6) & 7};
        local = {local.x + (axis == 0 ? (sign > 0 ? 1 : 0) : 0), local.y + (axis == 1 ? (sign > 0 ? 1 : 0) : 0), local.z + (axis == 2 ? (sign > 0 ? 1 : 0) : 0)};
        float d[4];
        float3 corners[4];
        for (int q = 0; q < 4; ++q) {
            int a = (axis + 1) % 3, b = (axis + 2) % 3;
            float u = (q & 1 ? .5f : -.5f) * h, v = (q & 2 ? .5f : -.5f) * h;
            corners[q] = {p.x + (a == 0 ? u : 0) + (b == 0 ? v : 0), p.y + (a == 1 ? u : 0) + (b == 1 ? v : 0), p.z + (a == 2 ? u : 0) + (b == 2 ? v : 0)};
            int3 node{local.x + (a == 0 ? q & 1 : 0) + (b == 0 ? q >> 1 : 0), local.y + (a == 1 ? q & 1 : 0) + (b == 1 ? q >> 1 : 0), local.z + (a == 2 ? q & 1 : 0) + (b == 2 ? q >> 1 : 0)};
            d[q] = grid.distance[sample][(owner >> 9) * 729 + node.x + 9 * node.y + 81 * node.z];
        }
        float aperture = .5f * (triangle_fraction(d[0], d[1], d[3]) + triangle_fraction(d[0], d[3], d[2]));
        float3 first = solid_face_moment(d[0], d[1], d[3], corners[0], corners[1], corners[3]), second = solid_face_moment(d[0], d[3], d[2], corners[0], corners[3], corners[2]);
        float3 moment{(first.x + second.x) * .5f, (first.y + second.y) * .5f, (first.z + second.z) * .5f};
        float3 speed{};
        if (!grid.static_geometry && aperture < 1) {
            float inverse_covered = 1.0f / (1 - aperture);
            solid_distance({moment.x * inverse_covered, moment.y * inverse_covered, moment.z * inverse_covered}, time, colliders, count, &speed, nullptr, 4 * h);
        }
        face.area       = aperture * h * h;
        face.solid_flux = (1 - aperture) * h * h * (axis == 0 ? speed.x : axis == 1 ? speed.y : speed.z);
        bool left = face.left < 0 || grid.fluid[face.left], right = face.right < 0 || grid.fluid[face.right];
        face.weight   = left && right ? face.area / (face.distance * grid.layout.h) : 0;
        grid.faces[f] = face;
    }
    void geometry(Grid grid, const Collider* colliders, int count, float time, int sample, cudaStream_t stream) {
        if (!grid.layout.count) return;
        cell_geometry<<<grid.layout.count * 2, 256, 0, stream>>>(grid, colliders, count, time, sample);
        face_geometry<<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, colliders, count, time, sample);
    }
    __global__ void density_kernel(Grid grid, DensityBlock* output) {
        int b = blockIdx.x;
        if (!threadIdx.x) output[b].block = grid.layout.blocks[b];
        for (int i = threadIdx.x; i < 512; i += blockDim.x) output[b].values[i] = grid.state.smoke[b * 512 + i];
    }
    void pack_density(Grid grid, DensityBlock* output, cudaStream_t stream) {
        if (grid.layout.count) density_kernel<<<grid.layout.count, 256, 0, stream>>>(grid, output);
    }
} // namespace pyro::cuda::sparse
