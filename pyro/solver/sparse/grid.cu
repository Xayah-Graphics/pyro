#include "grid.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>

#include <geometry/clipping.cuh>
#include <solver/common/scene.cuh>

namespace pyro::sparse {
    __device__ float restriction(const float* field, int b, int3 p) {
        float sum = 0;
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) sum += field[b * 512 + 2 * p.x + x + 8 * (2 * p.y + y) + 64 * (2 * p.z + z)];
        return sum * .125f;
    }

    __device__ void root_box(Topology topology, GridConfig config, float3 low, float3 high) {
        float h = ldexpf(config.cell_size * 8, config.levels - 1);
        int3 l{int(floorf(low.x / h)) - config.pressure_padding, int(floorf(low.y / h)) - config.pressure_padding, int(floorf(low.z / h)) - config.pressure_padding};
        int3 u{int(floorf(high.x / h)) + config.pressure_padding, int(floorf(high.y / h)) + config.pressure_padding, int(floorf(high.z / h)) + config.pressure_padding};
        for (int z = l.z; z <= u.z; ++z)
            for (int y = l.y; y <= u.y; ++y)
                for (int x = l.x; x <= u.x; ++x)
                    if (insert(topology.roots, topology.table_size, key({x, y, z}, config.levels - 1)) < 0) atomicExch(topology.counters + 1, 1);
    }

    __global__ void hash_blocks(Layout layout) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= layout.count) return;
        auto block           = layout.blocks[b];
        int slot             = insert(layout.keys, layout.table_size, key(block.coordinate, block.level));
        layout.indices[slot] = b;
    }

    __global__ void classify(Grid grid, Topology topology, GridConfig config) {
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

    __global__ void request_old(Grid grid, Topology topology, GridConfig config) {
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

    __global__ void request_scene(Topology topology, GridConfig config, const SourceView* sources, int source_count, const ColliderView* colliders, int collider_count, float time, float horizon) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float3 low{}, high{};
        if (i < source_count) {
            SourceView source = sources[i];
            if (time + horizon < source.start_time || time > source.end_time) return;
            low  = source.instance.swept.lower;
            high = source.instance.swept.upper;
        } else if (i < source_count + collider_count) {
            const auto& body = colliders[i - source_count];
            float offset     = fabsf(body.surface_offset);
            low              = {body.instance.swept.lower.x - offset, body.instance.swept.lower.y - offset, body.instance.swept.lower.z - offset};
            high             = {body.instance.swept.upper.x + offset, body.instance.swept.upper.y + offset, body.instance.swept.upper.z + offset};
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

    __global__ void mark_refinement(Layout layout, Topology topology, GridConfig config, const SourceView* sources, int source_count, const ColliderView* colliders, int collider_count, float time, float horizon, float collider_speed) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= layout.count) return;
        auto block = layout.blocks[b];
        if (!block.level) return;
        float h = ldexpf(config.cell_size, block.level), width = h * 8;
        float3 center{(block.coordinate.x + .5f) * width, (block.coordinate.y + .5f) * width, (block.coordinate.z + .5f) * width};
        bool split = lookup(topology.refine, topology.table_size, key(block.coordinate, block.level)) >= 0;
        for (int i = 0; i < source_count && !split; ++i) {
            SourceView s = sources[i];
            if (time + horizon < s.start_time || time > s.end_time) continue;
            float support = width * 2.f + h * 8.f + s.instance.max_speed * horizon * 2;
            split |= source_weight(s, center, support, time + .5f * horizon) > 0;
        }
        if (collider_count) {
            float band = 7.f * h + collider_speed * horizon;
            float d    = solid_distance(center, time + .5f * horizon, colliders, collider_count, band, nullptr, nullptr);
            split |= fabsf(d) < band;
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

    template <bool reuse>
    __global__ void sample_corners(Grid grid, Layout previous, const float* previous_distance, const ColliderView* colliders, int count, float time, float* distance) {
        int b       = blockIdx.x;
        Block block = grid.layout.blocks[b];
        int old     = reuse ? find_block(previous, block.coordinate, block.level) : -1;
        float h = ldexpf(grid.layout.h, block.level), band = 4 * h;
        for (int local = threadIdx.x; local < 729; local += blockDim.x) {
            float value;
            if (old >= 0) value = previous_distance[old * 729 + local];
            else {
                float3 p{(block.coordinate.x * 8 + local % 9) * h, (block.coordinate.y * 8 + (local / 9) % 9) * h, (block.coordinate.z * 8 + local / 81) * h};
                value = solid_distance(p, time, colliders, count, band, nullptr, nullptr);
            }
            distance[b * 729 + local] = value;
        }
    }

    template <BoundaryMode boundary_mode>
    __global__ void cell_geometry(Grid grid, const ColliderView* colliders, int count, float time, const float* distance) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        if constexpr (boundary_mode == BoundaryMode::none) {
            grid.fluid[i]            = 1;
            grid.solid_divergence[i] = 0;
            return;
        }
        float h  = spacing(grid.layout, i);
        float3 p = center(grid.layout, i);
        int node = (i >> 9) * 729 + (i & 7) + 9 * ((i >> 3) & 7) + 81 * ((i >> 6) & 7);
        float d[8], maximum = -1.e30f;
        for (int k = 0; k < 8; ++k) {
            d[k]    = distance[node + (k & 1) + 9 * ((k >> 1) & 1) + 81 * (k >> 2)];
            maximum = fmaxf(maximum, d[k]);
        }
        grid.fluid[i]            = maximum > 0;
        grid.solid_divergence[i] = 0;
        if constexpr (boundary_mode == BoundaryMode::moving) {
            constexpr int tetrahedra[6][4]{{0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7}};
            float volume = 0;
            for (auto& t : tetrahedra) volume += ::geometry::tetrahedron_positive_fraction(d[t[0]], d[t[1]], d[t[2]], d[t[3]]) / 6.f;
            float rate = 0;
            if (volume < 1 && maximum > 0) solid_distance<false, true>(p, time, colliders, count, 4 * h, nullptr, &rate);
            grid.solid_divergence[i] = (1 - volume) * rate;
        }
    }

    template <BoundaryMode boundary_mode>
    __global__ void face_geometry(Grid grid, const ColliderView* colliders, int count, float time, const float* distance) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.left < 0 && face.right < 0) return;
        float3 p;
        float h;
        int axis, owner, sign;
        face_position(grid.layout, f, p, h, axis, owner, sign);
        if constexpr (boundary_mode == BoundaryMode::none) {
            face.area       = h * h;
            face.solid_flux = 0;
            face.weight     = face.area / (face.distance * grid.layout.h);
            grid.faces[f]   = face;
            return;
        }
        int3 local{owner & 7, (owner >> 3) & 7, (owner >> 6) & 7};
        local = {local.x + (axis == 0 ? (sign > 0 ? 1 : 0) : 0), local.y + (axis == 1 ? (sign > 0 ? 1 : 0) : 0), local.z + (axis == 2 ? (sign > 0 ? 1 : 0) : 0)};
        float d[4];
        float3 corners[4];
        for (int q = 0; q < 4; ++q) {
            int a = (axis + 1) % 3, b = (axis + 2) % 3;
            float u = (q & 1 ? .5f : -.5f) * h, v = (q & 2 ? .5f : -.5f) * h;
            corners[q] = {p.x + (a == 0 ? u : 0) + (b == 0 ? v : 0), p.y + (a == 1 ? u : 0) + (b == 1 ? v : 0), p.z + (a == 2 ? u : 0) + (b == 2 ? v : 0)};
            int3 node{local.x + (a == 0 ? q & 1 : 0) + (b == 0 ? q >> 1 : 0), local.y + (a == 1 ? q & 1 : 0) + (b == 1 ? q >> 1 : 0), local.z + (a == 2 ? q & 1 : 0) + (b == 2 ? q >> 1 : 0)};
            d[q] = distance[(owner >> 9) * 729 + node.x + 9 * node.y + 81 * node.z];
        }
        float aperture = .5f * (::geometry::triangle_positive_fraction(d[0], d[1], d[3]) + ::geometry::triangle_positive_fraction(d[0], d[3], d[2]));
        float3 speed{};
        if constexpr (boundary_mode == BoundaryMode::moving) {
            if (aperture < 1) {
                float3 first = ::geometry::triangle_negative_moment(d[0], d[1], d[3], corners[0], corners[1], corners[3]), second = ::geometry::triangle_negative_moment(d[0], d[3], d[2], corners[0], corners[3], corners[2]);
                float3 moment{(first.x + second.x) * .5f, (first.y + second.y) * .5f, (first.z + second.z) * .5f};
                float inverse_covered = 1.0f / (1 - aperture);
                solid_distance<true>({moment.x * inverse_covered, moment.y * inverse_covered, moment.z * inverse_covered}, time, colliders, count, 4 * h, &speed, nullptr);
            }
        }
        face.area       = aperture * h * h;
        face.solid_flux = (1 - aperture) * h * h * (axis == 0 ? speed.x : axis == 1 ? speed.y : speed.z);
        bool left = face.left < 0 || grid.fluid[face.left], right = face.right < 0 || grid.fluid[face.right];
        face.weight   = left && right ? face.area / (face.distance * grid.layout.h) : 0;
        grid.faces[f] = face;
    }

    __global__ void density_kernel(Grid grid, DensityBlock* output) {
        int b = blockIdx.x;
        if (!threadIdx.x) {
            auto block           = grid.layout.blocks[b];
            output[b].coordinate = {block.coordinate.x, block.coordinate.y, block.coordinate.z};
            output[b].level      = block.level;
            output[b].quiet      = block.quiet;
            output[b].remaining  = block.remaining;
        }
        for (int i = threadIdx.x; i < 512; i += blockDim.x) output[b].values[i] = grid.state.smoke[b * 512 + i];
    }

    unsigned long long scratch_bytes(Scratch scratch, ::cuda::stream_ref stream) {
        std::size_t sort_bytes = 0, unique_bytes = 0, reduce_bytes = 0;
        if (auto cuda_result = cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, scratch.keys[0], scratch.keys[1], scratch.values[0], scratch.values[1], scratch.capacity, 0, 64, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cub::DeviceSelect::Unique(nullptr, unique_bytes, scratch.keys[1], scratch.keys[2], scratch.count, scratch.capacity, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (auto cuda_result = cub::DeviceReduce::ReduceByKey(nullptr, reduce_bytes, scratch.keys[1], scratch.keys[2], scratch.values[1], scratch.values[0], scratch.count, ::cuda::std::plus<double>{}, scratch.capacity, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        return std::max({sort_bytes, unique_bytes, reduce_bytes});
    }

    void rebuild_hash(Layout layout, ::cuda::stream_ref stream) {
        ::cuda::fill_bytes(stream, ::cuda::std::span(layout.keys, static_cast<std::size_t>(layout.table_size)), 255);
        if (layout.count) ::cuda::launch(stream, ::cuda::distribute<256>(layout.count), hash_blocks, layout);
    }

    int adapt_topology(Grid old_grid, Grid new_grid, Topology topology, Scratch scratch, GridConfig config, const SourceView* sources, int source_count, const ColliderView* colliders, int collider_count, float time, float horizon, bool& changed, ::cuda::stream_ref stream) {
        ::cuda::fill_bytes(stream, ::cuda::std::span(topology.roots, static_cast<std::size_t>(topology.table_size)), 255);
        ::cuda::fill_bytes(stream, ::cuda::std::span(topology.refine, static_cast<std::size_t>(topology.table_size)), 255);
        ::cuda::fill_bytes(stream, ::cuda::std::span(topology.counters, static_cast<std::size_t>(2)), 0);
        if (old_grid.layout.count) {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(old_grid.layout.count), ::cuda::block_dims<256>()), classify, old_grid, topology, config);
            ::cuda::launch(stream, ::cuda::distribute<256>(old_grid.layout.count), request_old, old_grid, topology, config);
        }
        if (source_count + collider_count) ::cuda::launch(stream, ::cuda::distribute<256>(source_count + collider_count), request_scene, topology, config, sources, source_count, colliders, collider_count, time, horizon);
        ::cuda::launch(stream, ::cuda::distribute<256>(topology.table_size), collect_roots, topology, new_grid.layout.blocks);
        int status[2];
        ::cuda::copy_bytes(stream, ::cuda::std::span(topology.counters, 2), ::cuda::std::span(status, 2), {.src_access_order = ::cuda::source_access_order::stream});
        stream.sync();
        if (status[1]) return std::max(status[0], topology.capacity * 2);
        Layout layout = new_grid.layout;
        layout.count  = status[0];
        Block* output = topology.candidates;
        for (int pass = 0; pass < config.levels * 2 && layout.count; ++pass) {
            rebuild_hash(layout, stream);
            ::cuda::fill_bytes(stream, ::cuda::std::span(topology.split, static_cast<std::size_t>(layout.count)), 0);
            if (pass < config.levels) ::cuda::launch(stream, ::cuda::distribute<256>(layout.count), mark_refinement, layout, topology, config, sources, source_count, colliders, collider_count, time, horizon, old_grid.collider_speed);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(layout.count), ::cuda::block_dims<32>()), balance, layout, topology.split);
            ::cuda::fill_bytes(stream, ::cuda::std::span(topology.counters, static_cast<std::size_t>(2)), 0);
            ::cuda::launch(stream, ::cuda::distribute<256>(layout.count), split_blocks, layout, topology, output);
            ::cuda::copy_bytes(stream, ::cuda::std::span(topology.counters, 2), ::cuda::std::span(status, 2), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            if (status[1]) return status[0];
            std::swap(layout.blocks, output);
            bool done    = layout.count == status[0];
            layout.count = status[0];
            if (done) break;
        }
        ::cuda::fill_bytes(stream, ::cuda::std::span(topology.counters + 2, 1), 0);
        if (layout.count) {
            ::cuda::launch(stream, ::cuda::distribute<256>(layout.count), block_keys, layout, scratch.keys[0]);
            std::size_t bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceRadixSort::SortKeys(scratch.sort_storage, bytes, scratch.keys[0], scratch.keys[1], layout.count, 0, 64, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            new_grid.layout.count = layout.count;
            ::cuda::launch(stream, ::cuda::distribute<256>(layout.count), ordered_blocks, old_grid.layout, new_grid.layout, topology, scratch.keys[1]);
        }
        new_grid.layout.count = layout.count;
        rebuild_hash(new_grid.layout, stream);
        int different;
        ::cuda::copy_bytes(stream, ::cuda::std::span(topology.counters + 2, 1), ::cuda::std::span(&different, 1), {.src_access_order = ::cuda::source_access_order::stream});
        stream.sync();
        changed = different || layout.count != old_grid.layout.count;
        return layout.count;
    }

    void build_connectivity(Grid grid, ::cuda::stream_ref stream) {
        if (!grid.layout.count) return;
        ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), connect_faces, grid);
        ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 384), connect_links, grid);
    }

    void sample_geometry(Grid grid, Layout previous, const float* previous_distance, const ColliderView* colliders, int count, float time, int sample, ::cuda::stream_ref stream) {
        if (!grid.layout.count || grid.boundary_mode == BoundaryMode::none) return;
        if (previous_distance) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), sample_corners<true>, grid, previous, previous_distance, colliders, count, time, grid.distance[sample]);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), sample_corners<false>, grid, previous, previous_distance, colliders, count, time, grid.distance[sample]);
    }

    void update_geometry(Grid grid, const ColliderView* colliders, int count, float time, int sample, ::cuda::stream_ref stream) {
        if (!grid.layout.count) return;
        auto launch = [&]<BoundaryMode boundary_mode>() {
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count * 2), ::cuda::block_dims<256>()), cell_geometry<boundary_mode>, grid, colliders, count, time, grid.distance[sample]);
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), face_geometry<boundary_mode>, grid, colliders, count, time, grid.distance[sample]);
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none: launch.template operator()<BoundaryMode::none>(); break;
        case BoundaryMode::stationary: launch.template operator()<BoundaryMode::stationary>(); break;
        case BoundaryMode::moving: launch.template operator()<BoundaryMode::moving>(); break;
        }
    }

    void pack_density(Grid grid, DensityBlock* output, ::cuda::stream_ref stream) {
        if (grid.layout.count) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), density_kernel, grid, output);
    }

}
