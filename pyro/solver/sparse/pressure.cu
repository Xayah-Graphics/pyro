#include "pressure.cuh"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace pyro::sparse {
    constexpr float transfer_weight = .8f;
    constexpr float coarse_weight = .5f;
    constexpr int coarsest_iterations = 8;
    constexpr int coarse_cycle_limit = 1024;

    enum struct DotOperation { update_rz, denominator, initial_norm, residual_norm };

    __device__ int interface_edge(Grid grid, int group, int a, int b) {
        int low = min(a, b), high = max(a, b);
        return grid.layout.count * 1728 + group * 6 + low * 3 - low * (low - 1) / 2 + high - low - 1;
    }


    template <typename T>
    __device__ T block_sum(T value) {
        __shared__ T partial[8];
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = value;
        __syncthreads();
        value = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
        if (threadIdx.x < 32)
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
        return value;
    }

    template <typename T>
    __device__ T block_maximum(T value) {
        __shared__ T partial[8];
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) value = max(value, __shfl_down_sync(0xffffffffu, value, offset));
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = value;
        __syncthreads();
        value = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
        if (threadIdx.x < 32)
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) value = max(value, __shfl_down_sync(0xffffffffu, value, offset));
        return value;
    }

    template <bool adaptive>
    __device__ Edge physical_edge(Grid grid, int f) {
        int count = grid.layout.count * 1728;
        if (!adaptive || f < count) {
            Face face = grid.faces[f];
            return {face.left, face.right, face.weight};
        }
        int group = (f - count) / 6, pair = (f - count) % 6;
        Links links = grid.links[group];
        if (links.face[3] < 0) return {-1, -1, 0};
        constexpr int pairs[6][2]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
        Face a = grid.faces[links.face[pairs[pair][0]]], b = grid.faces[links.face[pairs[pair][1]]];
        int left    = spacing(grid.layout, a.left) < spacing(grid.layout, a.right) ? a.left : a.right;
        int right   = spacing(grid.layout, b.left) < spacing(grid.layout, b.right) ? b.left : b.right;
        float total = 0;
        for (int face : links.face) total += grid.faces[face].weight;
        return {left, right, total > 0 ? -a.weight * b.weight / total : 0};
    }

    template <bool fine, bool adaptive, typename T, bool regular = false>
    __device__ __forceinline__ T matrix_product(Grid grid, PressureLevel level, const T* x, int i) {
        if (level.diagonal[i] <= 0) return 0;
        if constexpr (fine && !regular)
            if (level.regular[i >> 9]) return matrix_product<true, adaptive, T, true>(grid, level, x, i);
        T result = level.boundary[i] * x[i];
        if constexpr (fine) {
            for (int a = 0; a < 3; ++a)
                for (int sign = -1; sign <= 1; sign += 2) {
                    int coordinate = (i >> (3 * a)) & 7, stride = 1 << (3 * a);
                    if (coordinate + sign >= 0 && coordinate + sign < 8) {
                        int f = negative_face(i + (sign > 0 ? stride : 0), a);
                        result += (regular ? ldexpf(1.f, grid.layout.blocks[i >> 9].level) : level.face_weights[f]) * (x[i] - x[i + sign * stride]);
                        continue;
                    }
                    for (int f : cell_faces(grid, i, a, sign).face)
                        if (f >= 0) {
                            Edge edge = level.edges[f];
                            if (edge.weight != 0) result += edge.weight * (x[i] - x[edge.left == i ? edge.right : edge.left]);
                            Face face = grid.faces[f];
                            int other = face.left == i ? face.right : face.left;
                            if constexpr (adaptive)
                                if (face.group >= 0 && spacing(grid.layout, i) < spacing(grid.layout, other)) {
                                    Links group = grid.links[face.group];
                                    int ordinal = 0;
                                    while (group.face[ordinal] != f) ++ordinal;
                                    for (int q = 0; q < 4; ++q)
                                        if (q != ordinal) {
                                            Edge peer = level.edges[interface_edge(grid, face.group, ordinal, q)];
                                            if (peer.weight != 0) result += peer.weight * (x[i] - x[peer.left == i ? peer.right : peer.left]);
                                        }
                                }
                        }
                }
        } else
            for (int k = 0; k < 24; ++k) {
                int j = level.adjacency[k * level.count + i];
                if (j < 0) break;
                result += level.adjacency_weights[k * level.count + i] * (x[i] - x[j]);
            }
        return result;
    }

    template <bool obstacles>
    __global__ void initialize_fine(Grid grid, PressureLevel fine, int* parents, int* open, unsigned char* anchors) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        if constexpr (obstacles) {
            parents[i] = i;
            open[i]    = 0;
            anchors[i] = 0;
        }
        fine.nodes[i] = {cell_coordinate(grid.layout, i), grid.layout.blocks[i >> 9].level};
    }

    __global__ void connect_components(Grid grid, int* parents) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.weight <= 0 || face.left < 0 || face.right < 0) return;
        int a = face.left, b = face.right;
        int parent = atomicAdd(parents + a, 0);
        while (parent != a) {
            int next = atomicAdd(parents + parent, 0);
            atomicMin(parents + a, next);
            a      = parent;
            parent = next;
        }
        parent = atomicAdd(parents + b, 0);
        while (parent != b) {
            int next = atomicAdd(parents + parent, 0);
            atomicMin(parents + b, next);
            b      = parent;
            parent = next;
        }
        while (a != b) {
            int high = max(a, b), low = min(a, b);
            unsigned peers = __match_any_sync(__activemask(), high);
            int target = __reduce_min_sync(peers, low), leader = __ffs(peers) - 1, old = 0;
            if ((threadIdx.x & 31) == leader) old = atomicCAS(parents + high, high, target);
            old = __shfl_sync(peers, old, leader);
            if (old == high && low == target) break;
            a      = old == high ? target : old;
            parent = atomicAdd(parents + a, 0);
            while (parent != a) {
                int next = atomicAdd(parents + parent, 0);
                atomicMin(parents + a, next);
                a      = parent;
                parent = next;
            }
            b      = low;
            parent = atomicAdd(parents + b, 0);
            while (parent != b) {
                int next = atomicAdd(parents + parent, 0);
                atomicMin(parents + b, next);
                b      = parent;
                parent = next;
            }
        }
    }

    __global__ void mark_open(Grid grid, int* parents, int* open) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        int root = i, parent = atomicAdd(parents + root, 0);
        while (parent != root) {
            int next = atomicAdd(parents + parent, 0);
            atomicMin(parents + root, next);
            root   = parent;
            parent = next;
        }
        atomicExch(parents + i, root);
        for (int a = 0; a < 3; ++a)
            for (int sign = -1; sign <= 1; sign += 2)
                for (int f : cell_faces(grid, i, a, sign).face)
                    if (f >= 0 && grid.faces[f].weight > 0 && (grid.faces[f].left < 0 || grid.faces[f].right < 0)) atomicExch(open + root, 1);
    }

    __global__ void mark_anchors(Grid grid, const int* parents, const int* open, unsigned char* anchors, IterationState* state) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= grid.layout.count * 512) return;
        anchors[i] = grid.fluid[i] && parents[i] == i && !open[i];
        if (anchors[i]) atomicAdd(&state->iterations, 1);
    }

    template <bool closed, bool adaptive>
    __global__ void fine_edges(Grid grid, PressureLevel fine, const unsigned char* anchors) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= fine.edge_count) return;
        Edge edge = physical_edge<adaptive>(grid, f);
        if (edge.left < 0 || edge.right < 0 || (closed && (anchors[edge.left] || anchors[edge.right]))) edge.weight = 0;
        fine.edges[f] = edge;
        if (f < grid.layout.count * 1728) fine.face_weights[f] = edge.weight;
    }

    template <bool closed, bool adaptive>
    __global__ void fine_diagonal(Grid grid, PressureLevel fine, const unsigned char* anchors) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        double diagonal = 0, boundary = 0;
        for (int a = 0; a < 3; ++a)
            for (int sign = -1; sign <= 1; sign += 2)
                for (int f : cell_faces(grid, i, a, sign).face)
                    if (f >= 0) {
                        Face face = grid.faces[f];
                        int j     = face.left == i ? face.right : face.left;
                        diagonal += face.weight;
                        if (j < 0 || (closed && anchors[j])) boundary += face.weight;
                        if constexpr (adaptive)
                            if (face.group >= 0 && spacing(grid.layout, i) < spacing(grid.layout, j)) {
                                Links group = grid.links[face.group];
                                int ordinal = 0;
                                while (group.face[ordinal] != f) ++ordinal;
                                for (int q = 0; q < 4; ++q)
                                    if (q != ordinal) {
                                        Edge edge    = physical_edge<adaptive>(grid, interface_edge(grid, face.group, ordinal, q));
                                        int neighbor = edge.left == i ? edge.right : edge.left;
                                        diagonal += edge.weight;
                                        if constexpr (closed)
                                            if (anchors[neighbor]) boundary += edge.weight;
                                    }
                            }
                    }
        fine.boundary[i] = closed && anchors[i] ? 1 : grid.fluid[i] ? float(boundary) : 0;
        fine.diagonal[i] = closed && anchors[i] ? 1 : grid.fluid[i] ? float(diagonal) : 0;
    }

    __global__ void classify_regular(Grid grid, PressureLevel fine) {
        int block = blockIdx.x;
        float weight = ldexpf(1.f, grid.layout.blocks[block].level);
        bool regular = true;
        for (int local = threadIdx.x; local < 512; local += blockDim.x) {
            int i = block * 512 + local;
            for (int axis = 0; axis < 3; ++axis)
                if (((local >> (3 * axis)) & 7) > 0) regular &= fine.face_weights[negative_face(i, axis)] == weight;
        }
        int uniform = __syncthreads_and(regular);
        if (!threadIdx.x) fine.regular[block] = uniform;
    }

    __global__ void parent_keys(PressureLevel fine, unsigned long long* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        Node n    = fine.nodes[i];
        keys[i]   = key({n.coordinate.x >> 1, n.coordinate.y >> 1, n.coordinate.z >> 1}, n.level + 1);
    }

    __global__ void unpack_nodes(PressureLevel coarse, const unsigned long long* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < coarse.count) coarse.nodes[i] = node(keys[i]);
    }

    __global__ void link_parents(PressureLevel fine, PressureLevel coarse, const unsigned long long* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        Node n    = fine.nodes[i];
        auto k    = key({n.coordinate.x >> 1, n.coordinate.y >> 1, n.coordinate.z >> 1}, n.level + 1);
        int low = 0, high = coarse.count;
        while (low < high) {
            int middle = (low + high) / 2;
            if (keys[middle] < k) low = middle + 1;
            else high = middle;
        }
        fine.parent[i]                   = low;
        int child                        = (n.coordinate.x & 1) + 2 * (n.coordinate.y & 1) + 4 * (n.coordinate.z & 1);
        coarse.children[low * 8 + child] = i;
    }

    __global__ void edge_keys(PressureLevel fine, unsigned long long* keys, float* values) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= fine.edge_count) return;
        Edge edge            = fine.edges[f];
        unsigned long long k = empty_key;
        float value          = 0;
        if (edge.weight != 0 && edge.left >= 0 && edge.right >= 0) {
            int a = fine.parent[edge.left], b = fine.parent[edge.right];
            if (a != b) {
                k     = (static_cast<unsigned long long>(min(a, b)) << 32) | unsigned(max(a, b));
                value = edge.weight;
            }
        }
        keys[f]   = k;
        values[f] = value;
    }

    __global__ void trim_empty_edges(const unsigned long long* keys, int* count) {
        if (*count && keys[*count - 1] == empty_key) --*count;
    }

    __global__ void reduce_edges(PressureLevel coarse, const unsigned long long* unique, const float* values) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= coarse.edge_count) return;
        auto k = unique[f];
        Edge edge{int(k >> 32), int(k & 0xffffffff), values[f] * coarse_weight};
        coarse.edges[f] = edge;
        for (int i : {edge.left, edge.right}) {
            bool inserted = false;
            for (int j = 0; j < 24; ++j)
                if (atomicCAS(coarse.adjacency + j * coarse.count + i, -1, f) == -1) {
                    inserted = true;
                    break;
                }
            if (!inserted) __trap();
        }
    }

    __global__ void coarse_diagonal(PressureLevel fine, PressureLevel coarse) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coarse.count) return;
        int row[24];
        for (int j = 0; j < 24; ++j) row[j] = coarse.adjacency[j * coarse.count + i];
        for (int j = 1; j < 24; ++j) {
            int value = row[j], k = j;
            if (value < 0) continue;
            while (k && (row[k - 1] < 0 || row[k - 1] > value)) {
                row[k] = row[k - 1];
                --k;
            }
            row[k] = value;
        }
        for (int j = 0; j < 24; ++j) {
            int f = row[j];
            Edge edge = f >= 0 ? coarse.edges[f] : Edge{};
            coarse.adjacency[j * coarse.count + i] = f >= 0 ? (edge.left == i ? edge.right : edge.left) : -1;
            coarse.adjacency_weights[j * coarse.count + i] = edge.weight;
        }
        double boundary = 0, diagonal = 0;
        for (int c = 0; c < 8; ++c) {
            int child = coarse.children[i * 8 + c];
            if (child >= 0) boundary += fine.boundary[child];
        }
        for (int j = 0; j < 24 && row[j] >= 0; ++j) diagonal += coarse.edges[row[j]].weight;
        coarse.boundary[i] = float(boundary * coarse_weight);
        coarse.diagonal[i] = float(boundary * coarse_weight + diagonal);
    }

    template <BoundaryMode boundary_mode, bool closed, bool warm>
    __global__ void rhs_kernel(Grid grid, Fields fields, PressureLevel fine, const unsigned char* anchors, double* solution, float* residual) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        float h     = spacing(grid.layout, i);
        double flux = boundary_mode == BoundaryMode::moving ? -double(grid.solid_divergence[i]) * h * h * h : 0;
        for (int a = 0; a < 3; ++a)
            for (int sign = -1; sign <= 1; sign += 2)
                for (int f : cell_faces(grid, i, a, sign).face)
                    if (f >= 0) flux += sign * (double(grid.faces[f].area) * fields.velocity[f] + (boundary_mode == BoundaryMode::moving ? grid.faces[f].solid_flux : 0));
        fine.rhs[i] = residual[i] = grid.fluid[i] && (!closed || !anchors[i]) ? -flux / grid.layout.h : 0;
        if constexpr (warm) solution[i] = grid.fluid[i] && (!closed || !anchors[i]) ? solution[i] : 0;
        else solution[i] = 0;
    }

    template <bool fine, bool adaptive, bool second, bool zero>
    __global__ void smooth(Grid grid, PressureLevel level, const float* rhs, const float* input, float* output, float weight, float beta, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= level.count) return;
        float diagonal = level.diagonal[i];
        if (diagonal <= 0) {
            output[i] = 0;
            return;
        }
        if constexpr (zero && !second) output[i] = weight * (rhs[i] / diagonal);
        else {
            float value    = input[i];
            float residual = (rhs[i] - matrix_product<fine, adaptive>(grid, level, input, i)) / diagonal;
            if constexpr (second) output[i] = value + weight * residual + beta * (value - (zero ? 0 : output[i]));
            else output[i] = value + weight * residual;
        }
    }

    template <bool fine, bool adaptive>
    __global__ void prepare_restriction(Grid grid, PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < level.count) level.temporary[i] = level.diagonal[i] > 0 ? (rhs[i] - matrix_product<fine, adaptive>(grid, level, level.solution, i)) / level.diagonal[i] : 0;
    }

    template <bool fine, bool adaptive>
    __global__ void apply_restriction(Grid grid, PressureLevel level, float* output, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < level.count) output[i] = level.diagonal[i] * level.temporary[i] - transfer_weight * matrix_product<fine, adaptive>(grid, level, level.temporary, i);
    }

    template <bool finest, bool adaptive>
    __global__ void restrict_residual(Grid grid, PressureLevel fine, PressureLevel coarse, const float* input, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coarse.count) return;
        double value = 0;
        for (int c = 0; c < 8; ++c) {
            int j = coarse.children[i * 8 + c];
            if (j >= 0) value += input[j];
        }
        coarse.rhs[i] = float(value);
    }

    __global__ void prolongate(PressureLevel fine, PressureLevel coarse, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count) fine.temporary[i] = coarse.solution[fine.parent[i]];
    }

    template <bool finest, bool adaptive>
    __global__ void smooth_correction(Grid grid, PressureLevel fine, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count && fine.diagonal[i] > 0) fine.solution[i] += fine.temporary[i] - transfer_weight * matrix_product<finest, adaptive>(grid, fine, fine.temporary, i) / fine.diagonal[i];
    }

    __global__ void solve_coarsest(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ float x[256], next[256];
        int i = threadIdx.x;
        x[i]  = 0;
        __syncthreads();
        for (int iteration = 0; iteration < coarsest_iterations; ++iteration) {
            if (i < level.count) {
                float value = level.boundary[i] * x[i];
                for (int k = 0; k < 24; ++k) {
                    int j = level.adjacency[k * level.count + i];
                    if (j < 0) break;
                    value += level.adjacency_weights[k * level.count + i] * (x[i] - x[j]);
                }
                next[i] = level.diagonal[i] > 0 ? x[i] + (2.f / 3) * (rhs[i] - value) / level.diagonal[i] : 0;
            }
            __syncthreads();
            if (i < level.count) x[i] = next[i];
            __syncthreads();
        }
        if (i < level.count) level.solution[i] = x[i];
    }

    __global__ void coarse_cycle(Grid grid, PressureView workspace, int first) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        for (int l = first; l < workspace.level_count; ++l) {
            auto level = workspace.levels[l];
            const float* rhs = level.rhs;
            if (l + 1 == workspace.level_count) {
                for (int i = threadIdx.x; i < level.count; i += blockDim.x) level.solution[i] = 0;
                __syncthreads();
                for (int iteration = 0; iteration < coarsest_iterations; ++iteration) {
                    for (int i = threadIdx.x; i < level.count; i += blockDim.x) level.temporary[i] = level.diagonal[i] > 0 ? level.solution[i] + (2.f / 3) * (rhs[i] - matrix_product<false, false>(grid, level, level.solution, i)) / level.diagonal[i] : 0;
                    __syncthreads();
                    for (int i = threadIdx.x; i < level.count; i += blockDim.x) level.solution[i] = level.temporary[i];
                    __syncthreads();
                }
            } else {
                float upper = grid.layout.levels > 1 ? 3.f : 2.f;
                float theta = .5f * (upper + .3f), delta = .5f * (upper - .3f);
                float alpha = 1.f / (theta - delta * delta / (2 * theta)), beta = alpha * delta * delta / (2 * theta);
                for (int i = threadIdx.x; i < level.count; i += blockDim.x) level.temporary[i] = level.diagonal[i] > 0 ? rhs[i] / (theta * level.diagonal[i]) : 0;
                __syncthreads();
                for (int i = threadIdx.x; i < level.count; i += blockDim.x) {
                    float value = level.temporary[i];
                    level.solution[i] = level.diagonal[i] > 0 ? value + alpha * (rhs[i] - matrix_product<false, false>(grid, level, level.temporary, i)) / level.diagonal[i] + beta * value : 0;
                }
                __syncthreads();
                for (int i = threadIdx.x; i < level.count; i += blockDim.x) level.temporary[i] = level.diagonal[i] > 0 ? (rhs[i] - matrix_product<false, false>(grid, level, level.solution, i)) / level.diagonal[i] : 0;
                __syncthreads();
                auto coarse = workspace.levels[l + 1];
                for (int i = threadIdx.x; i < coarse.count; i += blockDim.x) {
                    double value = 0;
                    for (int c = 0; c < 8; ++c) {
                        int j = coarse.children[i * 8 + c];
                        if (j >= 0) value += level.diagonal[j] * level.temporary[j] - transfer_weight * matrix_product<false, false>(grid, level, level.temporary, j);
                    }
                    coarse.rhs[i] = float(value);
                }
                __syncthreads();
            }
        }
        for (int l = workspace.level_count - 2; l >= first; --l) {
            auto level = workspace.levels[l], coarse = workspace.levels[l + 1];
            for (int i = threadIdx.x; i < level.count; i += blockDim.x) level.temporary[i] = coarse.solution[level.parent[i]];
            __syncthreads();
            for (int i = threadIdx.x; i < level.count; i += blockDim.x)
                if (level.diagonal[i] > 0) level.solution[i] += level.temporary[i] - transfer_weight * matrix_product<false, false>(grid, level, level.temporary, i) / level.diagonal[i];
            __syncthreads();
            float upper = grid.layout.levels > 1 ? 3.f : 2.f;
            float theta = .5f * (upper + .3f), delta = .5f * (upper - .3f);
            float alpha = 1.f / (theta - delta * delta / (2 * theta)), beta = alpha * delta * delta / (2 * theta);
            for (int i = threadIdx.x; i < level.count; i += blockDim.x) level.temporary[i] = level.diagonal[i] > 0 ? level.solution[i] + (level.rhs[i] - matrix_product<false, false>(grid, level, level.solution, i)) / (theta * level.diagonal[i]) : 0;
            __syncthreads();
            for (int i = threadIdx.x; i < level.count; i += blockDim.x) {
                float value = level.temporary[i];
                level.solution[i] = level.diagonal[i] > 0 ? value + alpha * (level.rhs[i] - matrix_product<false, false>(grid, level, level.temporary, i)) / level.diagonal[i] + beta * (value - level.solution[i]) : 0;
            }
            __syncthreads();
        }
    }

    __global__ void sum_closed_residual(int count, const float* residual, const int* components, const int* open, double* sums) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count && !open[components[i]]) {
            auto peers = cooperative_groups::labeled_partition(cooperative_groups::coalesced_threads(), components[i]);
            double sum = cooperative_groups::reduce(peers, double(residual[i]), cooperative_groups::plus<double>{});
            if (!peers.thread_rank()) atomicAdd(sums + components[i], sum);
        }
    }

    template <DotOperation operation>
    __global__ void finish_dot(double* partial, int count, IterationState* state, float relative_tolerance, float absolute_tolerance) {
        if (state->converged || state->failed) return;
        constexpr bool maximum = operation == DotOperation::initial_norm || operation == DotOperation::residual_norm;
        double sum = 0, largest = 0;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            sum += partial[i];
            if constexpr (maximum) largest = fmax(largest, partial[count + i]);
        }
        sum = block_sum(sum);
        if constexpr (maximum) largest = block_maximum(largest);
        if (threadIdx.x) return;
        if (!isfinite(sum)) {
            state->failed = 1;
            return;
        }
        if constexpr (operation == DotOperation::update_rz) {
            state->previous_rz = state->rz;
            state->rz          = sum;
        } else if constexpr (operation == DotOperation::denominator) {
            state->denominator = sum;
            if (sum <= 0.0) state->failed = 1;
        } else {
            if constexpr (operation == DotOperation::initial_norm) state->initial_norm = sum;
            state->max_residual = static_cast<float>(largest);
            bool relative       = sum <= state->initial_norm * relative_tolerance * relative_tolerance;
            state->converged    = largest <= absolute_tolerance && (relative || largest <= absolute_tolerance * 0.1f);
        }
    }

    template <bool maximum, bool closed = false>
    __global__ void dot_blocks(Grid grid, const float* a, const float* b, double* partial, const IterationState* state, const double* component_sums, const unsigned char* anchors) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x, count = grid.layout.count * 512;
        double sum    = i < count ? double(a[i]) * b[i] : 0;
        float largest = 0;
        if constexpr (maximum)
            if (i < count) {
                float h = spacing(grid.layout, i);
                largest = (closed && anchors[i] ? float(fabs(component_sums[i])) : fabsf(a[i])) * grid.layout.h / (h * h * h);
            }
        sum = block_sum(sum);
        if constexpr (maximum) largest = block_maximum(largest);
        if (!threadIdx.x) {
            partial[blockIdx.x] = sum;
            if constexpr (maximum) partial[gridDim.x + blockIdx.x] = largest;
        }
    }

    template <bool adaptive>
    __global__ void initial_product(Grid grid, PressureLevel fine, PressureView workspace) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        double value = matrix_product<true, adaptive>(grid, fine, workspace.solution, i);
        workspace.component_sums[i] = value;
        double cross = block_sum(double(fine.rhs[i]) * value);
        __syncthreads();
        double norm = block_sum(value * value);
        if (!threadIdx.x) {
            workspace.reductions[blockIdx.x] = cross;
            workspace.reductions[gridDim.x + blockIdx.x] = norm;
        }
    }

    __global__ void initial_scale(double* partial, int count, IterationState* state) {
        double cross = 0, norm = 0;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            cross += partial[i];
            norm += partial[count + i];
        }
        cross = block_sum(cross);
        __syncthreads();
        norm = block_sum(norm);
        if (!threadIdx.x) state->initial_scale = norm > 0 ? cross / norm : 0;
    }

    __global__ void initial_guess(PressureLevel fine, PressureView workspace) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        double scale = workspace.iteration->initial_scale;
        workspace.solution[i] *= scale;
        workspace.residual[i] = float(double(fine.rhs[i]) - scale * workspace.component_sums[i]);
    }

    template <bool first>
    __global__ void direction_kernel(PressureLevel fine, float* direction, const IterationState* state, int max_iterations, cudaGraphConditionalHandle condition) {
        if constexpr (!first)
            if (!blockIdx.x && !threadIdx.x) cudaGraphSetConditional(condition, !state->converged && !state->failed && state->iterations < max_iterations);
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count) direction[i] = fine.solution[i] + (first ? 0 : float(state->rz / state->previous_rz) * direction[i]);
    }

    template <bool residual, bool adaptive, typename T>
    __global__ void product_kernel(Grid grid, PressureLevel fine, const T* input, float* output, double* partial, const IterationState* state) {
        if (!residual && (state->converged || state->failed)) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float value = i < fine.count ? (residual ? fine.rhs[i] : 0) + (residual ? -1 : 1) * matrix_product<true, adaptive>(grid, fine, input, i) : 0;
        if (i < fine.count) output[i] = value;
        if constexpr (!residual) {
            double sum = block_sum(i < fine.count ? double(input[i]) * value : 0);
            if (!threadIdx.x) partial[blockIdx.x] = sum;
        }
    }

    template <bool closed>
    __global__ void update_kernel(Grid grid, double* solution, float* residual, const float* direction, const float* product, double* partial, IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (!i) ++state->iterations;

        float alpha = float(state->rz / state->denominator);
        solution[i] += (state->rz / state->denominator) * double(direction[i]);
        residual[i] -= alpha * product[i];
        if constexpr (!closed) {
            double sum = block_sum(double(residual[i]) * residual[i]);
            float h = spacing(grid.layout, i);
            float largest = block_maximum(fabsf(residual[i]) * grid.layout.h / (h * h * h));
            if (!threadIdx.x) {
                partial[blockIdx.x] = sum;
                partial[gridDim.x + blockIdx.x] = largest;
            }
        }
    }

    template <bool adaptive, bool closed>
    __global__ void finish_smoothing(Grid grid, PressureLevel level, const float* rhs, float alpha, float beta, const unsigned char* anchors, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        float diagonal = level.diagonal[i], value = 0;
        if (diagonal > 0) {
            float input = level.temporary[i];
            float residual = (rhs[i] - matrix_product<true, adaptive>(grid, level, level.temporary, i)) / diagonal;
            value = input + alpha * residual + beta * (input - level.solution[i]);
        }
        if constexpr (closed) if (anchors[i]) value = 0;
        level.solution[i] = value;
        double sum = block_sum(double(rhs[i]) * value);
        if (!threadIdx.x) partial[blockIdx.x] = sum;
    }

    template <BoundaryMode boundary_mode, bool adaptive>
    __global__ void project_faces(Grid grid, Fields fields, const double* pressure) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.left < 0 && face.right < 0) return;
        if (boundary_mode == BoundaryMode::none || face.weight > 0) {
            double difference = (face.left >= 0 ? pressure[face.left] : 0) - (face.right >= 0 ? pressure[face.right] : 0);
            if constexpr (adaptive)
                if (face.group >= 0) {
                    double average = 0, total = 0;
                    for (int peer : grid.links[face.group].face) {
                        Face adjacent = grid.faces[peer];
                        int fine      = spacing(grid.layout, adjacent.left) < spacing(grid.layout, adjacent.right) ? adjacent.left : adjacent.right;
                        average += adjacent.weight * pressure[fine];
                        total += adjacent.weight;
                    }
                    average /= total;
                    difference = spacing(grid.layout, face.left) < spacing(grid.layout, face.right) ? average - pressure[face.right] : pressure[face.left] - average;
                }
            fields.velocity[f] += float(difference / face.distance);
        } else if (face.area <= 0) {
            if constexpr (boundary_mode == BoundaryMode::moving) {
                float h            = ldexpf(grid.layout.h, grid.layout.blocks[f / 1728].level);
                fields.velocity[f] = face.solid_flux / (h * h);
            } else fields.velocity[f] = 0;
        }
    }

    template <bool adaptive>
    bool fine_matrix(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, ::cuda::stream_ref stream) {
        ::cuda::fill_bytes(stream, ::cuda::std::span(workspace.iteration, 1), 0);
        auto fine = levels[0];
        if (grid.boundary_mode == BoundaryMode::none) ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), initialize_fine<false>, grid, fine, workspace.components, workspace.open, workspace.anchors);
        else ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), initialize_fine<true>, grid, fine, workspace.components, workspace.open, workspace.anchors);
        if (grid.boundary_mode != BoundaryMode::none) {
            ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), connect_components, grid, workspace.components);
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), mark_open, grid, workspace.components, workspace.open);
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), mark_anchors, grid, workspace.components, workspace.open, workspace.anchors, workspace.iteration);
        }
        int closed = 0;
        if (grid.boundary_mode != BoundaryMode::none) {
            ::cuda::copy_bytes(stream, ::cuda::std::span(&workspace.iteration->iterations, 1), ::cuda::std::span(&closed, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
        }
        if (closed) {
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.edge_count), fine_edges<true, adaptive>, grid, fine, workspace.anchors);
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), fine_diagonal<true, adaptive>, grid, fine, workspace.anchors);
        } else {
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.edge_count), fine_edges<false, adaptive>, grid, fine, workspace.anchors);
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), fine_diagonal<false, adaptive>, grid, fine, workspace.anchors);
        }
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.count), ::cuda::block_dims<256>()), classify_regular, grid, fine);
        return closed != 0;
    }

    int coarse_nodes(PressureLevel fine, PressureLevel coarse, Scratch scratch, ::cuda::stream_ref stream) {
        int count = coarse.count;
        if (!coarse.nodes) {
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), parent_keys, fine, scratch.keys[0]);
            std::size_t bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceRadixSort::SortKeys(scratch.sort_storage, bytes, scratch.keys[0], scratch.keys[1], fine.count, 0, 64, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceSelect::Unique(scratch.sort_storage, bytes, scratch.keys[1], scratch.keys[2], scratch.count, fine.count, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            ::cuda::copy_bytes(stream, ::cuda::std::span(scratch.count, 1), ::cuda::std::span(&count, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            return count;
        }
        coarse.count = count;
        ::cuda::fill_bytes(stream, ::cuda::std::span(coarse.children, static_cast<std::size_t>(count * 8ull)), 255);
        ::cuda::launch(stream, ::cuda::distribute<256>(count), unpack_nodes, coarse, scratch.keys[2]);
        ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), link_parents, fine, coarse, scratch.keys[2]);
        return count;
    }

    int coarse_edges(PressureLevel fine, PressureLevel coarse, Scratch scratch, ::cuda::stream_ref stream) {
        int count = 0;
        if (fine.edge_count) {
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.edge_count), edge_keys, fine, scratch.keys[0], scratch.values[0]);
            std::size_t bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceRadixSort::SortPairs(scratch.sort_storage, bytes, scratch.keys[0], scratch.keys[1], scratch.values[0], scratch.values[1], fine.edge_count, 0, 64, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceReduce::ReduceByKey(scratch.sort_storage, bytes, scratch.keys[1], scratch.keys[2], scratch.values[1], scratch.values[0], scratch.count, ::cuda::std::plus<double>{}, fine.edge_count, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<1>()), trim_empty_edges, scratch.keys[2], scratch.count);
            ::cuda::copy_bytes(stream, ::cuda::std::span(scratch.count, 1), ::cuda::std::span(&count, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
        }
        coarse.edge_count = count;
        ::cuda::fill_bytes(stream, ::cuda::std::span(coarse.adjacency, static_cast<std::size_t>(coarse.count * 24ull)), 255);
        if (count) ::cuda::launch(stream, ::cuda::distribute<256>(count), reduce_edges, coarse, scratch.keys[2], scratch.values[0]);
        ::cuda::launch(stream, ::cuda::distribute<256>(coarse.count), coarse_diagonal, fine, coarse);
        return count;
    }

    void assemble_rhs(std::span<const PressureLevel> levels, Grid grid, Fields fields, PressureView workspace, bool warm, ::cuda::stream_ref stream) {
        auto launch = [&]<BoundaryMode boundary_mode>() {
            auto dispatch = [&]<bool reuse>() {
            if (workspace.closed) ::cuda::launch(stream, ::cuda::distribute<256>(levels[0].count), rhs_kernel<boundary_mode, true, reuse>, grid, fields, levels[0], workspace.anchors, workspace.solution, workspace.residual);
            else ::cuda::launch(stream, ::cuda::distribute<256>(levels[0].count), rhs_kernel<boundary_mode, false, reuse>, grid, fields, levels[0], workspace.anchors, workspace.solution, workspace.residual);
            };
            if (warm) dispatch.template operator()<true>();
            else dispatch.template operator()<false>();
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none: launch.template operator()<BoundaryMode::none>(); break;
        case BoundaryMode::stationary: launch.template operator()<BoundaryMode::stationary>(); break;
        case BoundaryMode::moving: launch.template operator()<BoundaryMode::moving>(); break;
        }
    }

    template <bool adaptive>
    void precondition(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, ::cuda::stream_ref stream) {
        for (int l = 0; l < workspace.level_count; ++l) {
            auto level       = levels[l];
            const float* rhs = l ? level.rhs : workspace.residual;
            if (l >= 1 && l + 1 < workspace.level_count && level.count <= coarse_cycle_limit) {
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), coarse_cycle, grid, workspace, l);
                break;
            }
            if (l + 1 == workspace.level_count) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), solve_coarsest, level, rhs, workspace.iteration);
            else {
                float upper = grid.layout.levels > 1 ? 3.f : 2.f;
                float theta = .5f * (upper + .3f), delta = .5f * (upper - .3f);
                float alpha = 1.f / (theta - delta * delta / (2 * theta)), beta = alpha * delta * delta / (2 * theta);
                auto coarse = levels[l + 1];
                auto launch = [&]<bool fine>() {
                    ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < fine, fine && adaptive, false, true >, grid, level, rhs, level.solution, level.temporary, 1.f / theta, 0, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < fine, fine && adaptive, true, true >, grid, level, rhs, level.temporary, level.solution, alpha, beta, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::distribute<256>(level.count), prepare_restriction<fine, fine && adaptive>, grid, level, rhs, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::distribute<256>(level.count), apply_restriction<fine, fine && adaptive>, grid, level, workspace.product, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::distribute<256>(coarse.count), restrict_residual < fine, fine && adaptive >, grid, level, coarse, workspace.product, workspace.iteration);
                };
                if (l == 0) launch.template operator()<true>();
                else launch.template operator()<false>();
            }
        }
        for (int l = workspace.level_count - 2; l >= 0; --l) {
            auto level       = levels[l];
            if (l >= 1 && level.count <= coarse_cycle_limit) continue;
            const float* rhs = l ? level.rhs : workspace.residual;
            float upper      = grid.layout.levels > 1 ? 3.f : 2.f;
            float theta = .5f * (upper + .3f), delta = .5f * (upper - .3f);
            float alpha = 1.f / (theta - delta * delta / (2 * theta)), beta = alpha * delta * delta / (2 * theta);
            ::cuda::launch(stream, ::cuda::distribute<256>(level.count), prolongate, level, levels[l + 1], workspace.iteration);
            auto launch = [&]<bool fine>() {
                ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth_correction<fine, fine && adaptive>, grid, level, workspace.iteration);
                ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < fine, fine && adaptive, false, false >, grid, level, rhs, level.solution, level.temporary, 1.f / theta, 0, workspace.iteration);
                if constexpr (fine) {
                    if (workspace.closed) ::cuda::launch(stream, ::cuda::distribute<256>(level.count), finish_smoothing<adaptive, true>, grid, level, rhs, alpha, beta, workspace.anchors, workspace.reductions, workspace.iteration);
                    else ::cuda::launch(stream, ::cuda::distribute<256>(level.count), finish_smoothing<adaptive, false>, grid, level, rhs, alpha, beta, workspace.anchors, workspace.reductions, workspace.iteration);
                }
                else ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < false, false, true, false >, grid, level, rhs, level.temporary, level.solution, alpha, beta, workspace.iteration);
            };
            if (l == 0) launch.template operator()<true>();
            else launch.template operator()<false>();
        }
    }

    void component_sums(Grid grid, PressureView workspace, ::cuda::stream_ref stream) {
        if (!workspace.closed) return;
        int count = grid.layout.count * 512;
        ::cuda::fill_bytes(stream, ::cuda::std::span(workspace.component_sums, static_cast<std::size_t>(count)), 0);
        ::cuda::launch(stream, ::cuda::distribute<256>(count), sum_closed_residual, count, workspace.residual, workspace.components, workspace.open, workspace.component_sums);
    }

    template <bool adaptive>
    void begin_pressure(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        int count = (fine.count + 255) / 256;
        component_sums(grid, workspace, stream);
        if (workspace.closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<true, true>, grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, workspace.component_sums, workspace.anchors);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<true>, grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::initial_norm>, workspace.reductions, count, workspace.iteration, config.relative_tolerance, config.divergence_tolerance * .25f);
        precondition<adaptive>(levels, grid, workspace, stream);
        if (workspace.level_count == 1) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<false>, grid, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::update_rz>, workspace.reductions, count, workspace.iteration, 0, 0);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), direction_kernel<true>, fine, workspace.direction, workspace.iteration, config.max_iterations, cudaGraphConditionalHandle{});
    }

    template <bool adaptive>
    void iterate_pressure(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream, cudaGraphConditionalHandle condition) {
        auto fine = levels[0];
        int count = (fine.count + 255) / 256;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), product_kernel<false, adaptive, float>, grid, fine, workspace.direction, workspace.product, workspace.reductions, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::denominator>, workspace.reductions, count, workspace.iteration, 0, 0);
        if (workspace.closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), update_kernel<true>, grid, workspace.solution, workspace.residual, workspace.direction, workspace.product, workspace.reductions, workspace.iteration);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), update_kernel<false>, grid, workspace.solution, workspace.residual, workspace.direction, workspace.product, workspace.reductions, workspace.iteration);
        component_sums(grid, workspace, stream);
        if (workspace.closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<true, true>, grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, workspace.component_sums, workspace.anchors);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::residual_norm>, workspace.reductions, count, workspace.iteration, config.relative_tolerance, config.divergence_tolerance * .25f);
        precondition<adaptive>(levels, grid, workspace, stream);
        if (workspace.level_count == 1) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<false>, grid, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::update_rz>, workspace.reductions, count, workspace.iteration, 0, 0);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), direction_kernel<false>, fine, workspace.direction, workspace.iteration, config.max_iterations, condition);
    }

    template <bool adaptive>
    void check_pressure(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        int count = (fine.count + 255) / 256;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), product_kernel<true, adaptive, double>, grid, fine, workspace.solution, workspace.residual, workspace.reductions, workspace.iteration);
        component_sums(grid, workspace, stream);
        if (workspace.closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<true, true>, grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, workspace.component_sums, workspace.anchors);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<true>, grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::residual_norm>, workspace.reductions, count, workspace.iteration, config.relative_tolerance, config.divergence_tolerance * .5f);
    }

    void apply_pressure(Grid grid, Fields fields, const double* pressure, ::cuda::stream_ref stream) {
        auto launch = [&]<BoundaryMode boundary_mode>() {
            if (grid.layout.levels > 1) ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), project_faces<boundary_mode, true>, grid, fields, pressure);
            else ::cuda::launch(stream, ::cuda::distribute<256>(grid.layout.count * 1728), project_faces<boundary_mode, false>, grid, fields, pressure);
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none: launch.template operator()<BoundaryMode::none>(); break;
        case BoundaryMode::stationary: launch.template operator()<BoundaryMode::stationary>(); break;
        case BoundaryMode::moving: launch.template operator()<BoundaryMode::moving>(); break;
        }
    }

    Pressure::Pressure(PressureConfig settings, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool)
        : config(settings),
          stream(stream),
          pool(pool),
          iteration_graph(nullptr, cudaGraphExecDestroy),
          iteration_storage(::cuda::make_buffer<IterationState>(stream, pool, 1, ::cuda::no_init)),
          level_storage(::cuda::make_buffer<PressureLevel>(stream, pool, 32, ::cuda::no_init)),
          node_storage(stream, pool),
          edge_storage(stream, pool),
          index_storage(stream, pool),
          anchor_storage(stream, pool),
          field_storage(stream, pool),
          double_storage(stream, pool) {
        view.iteration = iteration_storage.data();
        view.levels = level_storage.data();
        ::cuda::fill_bytes(stream, iteration_storage, 0);
        for (int i = 0; i < 31; ++i) {
            coarse_node_storage.emplace_back(stream, pool);
            coarse_edge_storage.emplace_back(stream, pool);
            coarse_index_storage.emplace_back(stream, pool);
            coarse_field_storage.emplace_back(stream, pool);
        }
    }

    void Pressure::reserve(Layout layout) {
        iteration_graph.reset();
        matrix_dirty = topology_dirty = true;
        solution_valid = false;
        auto cells = std::size_t(layout.capacity) * 512, faces = std::size_t(layout.capacity) * 1728;
        auto edges     = std::size_t(layout.capacity) * (layout.levels > 1 ? 4032 : 1728);
        node_storage   = ::cuda::make_buffer<Node>(stream, pool, cells, ::cuda::no_init);
        edge_storage   = ::cuda::make_buffer<Edge>(stream, pool, edges, ::cuda::no_init);
        index_storage  = ::cuda::make_buffer<int>(stream, pool, cells * 3, ::cuda::no_init);
        anchor_storage = ::cuda::make_buffer<unsigned char>(stream, pool, cells + layout.capacity, ::cuda::no_init);
        field_storage  = ::cuda::make_buffer<float>(stream, pool, cells * 8 + faces, ::cuda::no_init);
        double_storage = ::cuda::make_buffer<double>(stream, pool, cells * 2 + (layout.capacity + 1ull) * 6, ::cuda::no_init);
        ::cuda::fill_bytes(stream, node_storage, 0);
        ::cuda::fill_bytes(stream, edge_storage, 0);
        ::cuda::fill_bytes(stream, index_storage, 0);
        ::cuda::fill_bytes(stream, anchor_storage, 0);
        ::cuda::fill_bytes(stream, field_storage, 0);
        ::cuda::fill_bytes(stream, double_storage, 0);
        auto& fine          = hierarchy[0];
        fine.nodes          = node_storage.data();
        fine.edges          = edge_storage.data();
        fine.parent         = index_storage.data();
        fine.face_weights   = field_storage.data();
        fine.regular        = anchor_storage.data() + cells;
        auto* field         = field_storage.data() + faces;
        fine.boundary       = field;
        fine.diagonal       = field + cells;
        fine.rhs            = field + cells * 2;
        fine.solution       = field + cells * 3;
        fine.temporary      = field + cells * 4;
        fine.capacity       = static_cast<int>(cells);
        view.residual       = field + cells * 5;
        view.direction      = field + cells * 6;
        view.product        = field + cells * 7;
        view.solution       = double_storage.data();
        view.component_sums = double_storage.data() + cells;
        view.reductions     = double_storage.data() + cells * 2;
        view.components     = index_storage.data() + cells;
        view.open           = index_storage.data() + cells * 2;
        view.anchors        = anchor_storage.data();
    }

    void Pressure::build(Grid grid, Scratch scratch) {
        if (!grid.layout.count) return;
        ++matrix_build_count;
        topology_build_count += topology_dirty;
        auto levels = std::span(hierarchy);
        if (topology_dirty) {
            iteration_graph.reset();
            solution_valid = false;
        }
        bool was_closed = view.closed;
        auto& fine      = hierarchy[0];
        fine.count      = grid.layout.count * 512;
        fine.edge_count = grid.layout.count * (grid.layout.levels > 1 ? 4032 : 1728);
        if (grid.layout.levels > 1) view.closed = fine_matrix<true>(levels, grid, view, stream);
        else view.closed = fine_matrix<false>(levels, grid, view, stream);
        if (was_closed != view.closed) iteration_graph.reset();
        view.level_count = 1;
        for (int depth = 1; hierarchy[depth - 1].count > 128; ++depth) {
            auto& previous = hierarchy[depth - 1];
            auto& level    = hierarchy[depth];
            int count      = topology_dirty ? coarse_nodes(previous, {}, scratch, stream) : level.count;
            if (count > level.capacity) {
                int capacity  = (count + 63) / 64 * 64;
                auto& nodes   = coarse_node_storage[depth - 1];
                auto& edges   = coarse_edge_storage[depth - 1];
                auto& indices = coarse_index_storage[depth - 1];
                auto& fields  = coarse_field_storage[depth - 1];
                nodes         = ::cuda::make_buffer<Node>(stream, pool, capacity, ::cuda::no_init);
                edges         = ::cuda::make_buffer<Edge>(stream, pool, capacity * 12ull, ::cuda::no_init);
                indices       = ::cuda::make_buffer<int>(stream, pool, capacity * 33ull, ::cuda::no_init);
                fields        = ::cuda::make_buffer<float>(stream, pool, capacity * 29ull, ::cuda::no_init);
                ::cuda::fill_bytes(stream, nodes, 0);
                ::cuda::fill_bytes(stream, edges, 0);
                ::cuda::fill_bytes(stream, indices, 0);
                ::cuda::fill_bytes(stream, fields, 0);
                level.capacity = capacity;

                level.nodes     = nodes.data();
                level.edges     = edges.data();
                level.parent    = indices.data();
                level.children  = indices.data() + capacity;
                level.adjacency = indices.data() + capacity * 9ull;
                level.boundary  = fields.data();
                level.diagonal  = fields.data() + capacity;
                level.rhs       = fields.data() + capacity * 2ull;
                level.solution  = fields.data() + capacity * 3ull;
                level.temporary = fields.data() + capacity * 4ull;
                level.adjacency_weights = fields.data() + capacity * 5ull;
            }
            level.count = count;
            if (topology_dirty) coarse_nodes(previous, level, scratch, stream);
            level.edge_count = coarse_edges(previous, level, scratch, stream);
            ++view.level_count;
        }
        ::cuda::copy_bytes(stream, ::cuda::std::span(hierarchy.data(), static_cast<std::size_t>(view.level_count)), level_storage.first(view.level_count), {.src_access_order = ::cuda::source_access_order::during_api_call});
        matrix_dirty = topology_dirty = false;
    }

    int Pressure::project(Grid grid, Fields fields, Scratch scratch) {
        ++projection_count;
        if (matrix_dirty) build(grid, scratch);
        auto levels = std::span(hierarchy).first(view.level_count);
        assemble_rhs(levels, grid, fields, view, solution_valid, stream);
        ::cuda::fill_bytes(stream, ::cuda::std::span(view.iteration, 1), 0);
        if (solution_valid) {
            int count = grid.layout.count * 2;
            if (grid.layout.levels > 1) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), initial_product<true>, grid, levels[0], view);
            else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), initial_product<false>, grid, levels[0], view);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), initial_scale, view.reductions, count, view.iteration);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), initial_guess, levels[0], view);
        }
        if (grid.layout.levels > 1) begin_pressure<true>(levels, grid, view, config, stream);
        else begin_pressure<false>(levels, grid, view, config, stream);
        if (!iteration_graph) {
            cudaGraph_t captured{};
            if (auto result = cudaGraphCreate(&captured, 0); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph{captured, cudaGraphDestroy};
            cudaGraphConditionalHandle condition{};
            if (auto result = cudaGraphConditionalHandleCreate(&condition, graph.get(), 1, cudaGraphCondAssignDefault); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            cudaGraphNodeParams parameters{};
            parameters.type               = cudaGraphNodeTypeConditional;
            parameters.conditional.handle = condition;
            parameters.conditional.type   = cudaGraphCondTypeWhile;
            parameters.conditional.size   = 1;
            cudaGraphNode_t loop{};
            if (auto result = cudaGraphAddNode(&loop, graph.get(), nullptr, nullptr, 0, &parameters); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            auto body = parameters.conditional.phGraph_out[0];
            if (auto result = cudaStreamBeginCaptureToGraph(stream.get(), body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            if (grid.layout.levels > 1) iterate_pressure<true>(levels, grid, view, config, stream, condition);
            else iterate_pressure<false>(levels, grid, view, config, stream, condition);
            if (auto result = cudaStreamEndCapture(stream.get(), &body); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            cudaGraphExec_t executable{};
            if (auto result = cudaGraphInstantiate(&executable, graph.get(), 0); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            iteration_graph.reset(executable);
            ++graph_build_count;
        }
        IterationState status{};
        for (;;) {
            if (auto cuda_result = cudaGraphLaunch(iteration_graph.get(), stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            ::cuda::copy_bytes(stream, ::cuda::std::span(view.iteration, 1), ::cuda::std::span(&status, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            if (status.failed) break;
            if (status.converged || status.iterations >= config.max_iterations) {
                status.converged = 0;
                ::cuda::copy_bytes(stream, ::cuda::std::span(&status, 1), ::cuda::std::span(view.iteration, 1), {.src_access_order = ::cuda::source_access_order::stream});
                if (grid.layout.levels > 1) check_pressure<true>(levels, grid, view, config, stream);
                else check_pressure<false>(levels, grid, view, config, stream);
                ::cuda::copy_bytes(stream, ::cuda::std::span(view.iteration, 1), ::cuda::std::span(&status, 1), {.src_access_order = ::cuda::source_access_order::stream});
                stream.sync();
                if (status.max_residual <= config.divergence_tolerance * .5f || status.failed || status.iterations >= config.max_iterations) break;
                status.converged = 0;
                ::cuda::copy_bytes(stream, ::cuda::std::span(&status, 1), ::cuda::std::span(view.iteration, 1), {.src_access_order = ::cuda::source_access_order::stream});
                if (grid.layout.levels > 1) begin_pressure<true>(levels, grid, view, config, stream);
                else begin_pressure<false>(levels, grid, view, config, stream);
            }
        }
        if (status.failed || status.max_residual > config.divergence_tolerance * .5f) throw std::runtime_error("Sparse pressure solve failed: iterations=" + std::to_string(status.iterations) + ", true divergence residual=" + std::to_string(status.max_residual));
        apply_pressure(grid, fields, view.solution, stream);
        solution_valid = true;
        return status.iterations;
    }

}
