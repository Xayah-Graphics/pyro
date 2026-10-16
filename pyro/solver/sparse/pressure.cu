#include "pressure.cuh"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>

namespace pyro::sparse {
    enum struct DotOperation { update_rz, denominator, initial_norm, residual_norm };

    __device__ int interface_edge(Grid grid, int group, int a, int b) {
        int low = min(a, b), high = max(a, b);
        return grid.layout.count * 1728 + group * 6 + low * 3 - low * (low - 1) / 2 + high - low - 1;
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

    template <bool fine, bool adaptive, typename T>
    __device__ T matrix_product(Grid grid, PressureLevel level, const T* x, int i) {
        if (level.diagonal[i] <= 0) return 0;
        T result = level.boundary[i] * x[i];
        if constexpr (fine) {
            for (int a = 0; a < 3; ++a)
                for (int sign = -1; sign <= 1; sign += 2) {
                    int coordinate = (i >> (3 * a)) & 7, stride = 1 << (3 * a);
                    if (coordinate + sign >= 0 && coordinate + sign < 8) {
                        int f = negative_face(i + (sign > 0 ? stride : 0), a);
                        result += level.face_weights[f] * (x[i] - x[i + sign * stride]);
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
                int f = level.adjacency[k * level.count + i];
                if (f < 0) break;
                Edge edge = level.edges[f];
                result += edge.weight * (x[i] - x[edge.left == i ? edge.right : edge.left]);
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
            int old = atomicCAS(parents + high, high, low);
            if (old == high) break;
            a      = old;
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

    __global__ void parent_keys(PressureLevel fine, unsigned long long* keys, int depth) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        Node n    = fine.nodes[i];
        int shift = max(0, depth - n.level);
        keys[i]   = key({n.coordinate.x >> shift, n.coordinate.y >> shift, n.coordinate.z >> shift}, max(depth, n.level));
    }

    __global__ void unpack_nodes(PressureLevel coarse, const unsigned long long* keys) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < coarse.count) coarse.nodes[i] = node(keys[i]);
    }

    __global__ void link_parents(PressureLevel fine, PressureLevel coarse, const unsigned long long* keys, int depth) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        Node n    = fine.nodes[i];
        int shift = max(0, depth - n.level);
        auto k    = key({n.coordinate.x >> shift, n.coordinate.y >> shift, n.coordinate.z >> shift}, max(depth, n.level));
        int low = 0, high = coarse.count;
        while (low < high) {
            int middle = (low + high) / 2;
            if (keys[middle] < k) low = middle + 1;
            else high = middle;
        }
        fine.parent[i]                   = low;
        int child                        = shift ? (n.coordinate.x & 1) + 2 * (n.coordinate.y & 1) + 4 * (n.coordinate.z & 1) : 0;
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

    __global__ void reduce_edges(PressureLevel coarse, const unsigned long long* sorted, const float* values, const unsigned long long* unique, int count) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= coarse.edge_count) return;
        auto k  = unique[f];
        int low = 0, high = count;
        while (low < high) {
            int middle = (low + high) / 2;
            if (sorted[middle] < k) low = middle + 1;
            else high = middle;
        }
        double sum = 0;
        for (int j = low; j < count && sorted[j] == k; ++j) sum += values[j];
        Edge edge{int(k >> 32), int(k & 0xffffffff), float(sum)};
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
        for (int j = 0; j < 24; ++j) coarse.adjacency[j * coarse.count + i] = row[j];
        double boundary = 0, diagonal = 0;
        for (int c = 0; c < 8; ++c) {
            int child = coarse.children[i * 8 + c];
            if (child >= 0) boundary += fine.boundary[child];
        }
        for (int j = 0; j < 24 && row[j] >= 0; ++j) diagonal += coarse.edges[row[j]].weight;
        coarse.boundary[i] = float(boundary);
        coarse.diagonal[i] = float(boundary + diagonal);
    }

    template <BoundaryMode boundary_mode, bool closed>
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
        solution[i]               = 0;
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

    template <bool finest, bool adaptive>
    __global__ void restrict_residual(Grid grid, PressureLevel fine, PressureLevel coarse, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coarse.count) return;
        double value = 0;
        for (int c = 0; c < 8; ++c) {
            int j = coarse.children[i * 8 + c];
            if (j >= 0) value += rhs[j] - matrix_product<finest, adaptive>(grid, fine, fine.solution, j);
        }
        coarse.rhs[i] = float(value);
    }

    __global__ void prolongate(PressureLevel fine, PressureLevel coarse, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count) fine.solution[i] += coarse.solution[fine.parent[i]];
    }

    __global__ void solve_coarsest(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ float x[256], next[256];
        int i = threadIdx.x;
        x[i]  = 0;
        __syncthreads();
        for (int iteration = 0; iteration < 96; ++iteration) {
            if (i < level.count) {
                float value = level.boundary[i] * x[i];
                for (int k = 0; k < 24; ++k) {
                    int f = level.adjacency[k * level.count + i];
                    if (f < 0) break;
                    Edge edge = level.edges[f];
                    value += edge.weight * (x[i] - x[edge.left == i ? edge.right : edge.left]);
                }
                next[i] = level.diagonal[i] > 0 ? x[i] + (2.f / 3) * (rhs[i] - value) / level.diagonal[i] : 0;
            }
            __syncthreads();
            if (i < level.count) x[i] = next[i];
            __syncthreads();
        }
        if (i < level.count) level.solution[i] = x[i];
    }

    __global__ void sum_closed_residual(int count, const float* residual, const int* components, const int* open, double* sums) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count && !open[components[i]]) atomicAdd(sums + components[i], double(residual[i]));
    }

    template <DotOperation operation>
    __global__ void finish_dot(double* partial, int count, IterationState* state, float relative_tolerance, float absolute_tolerance) {
        if (state->converged || state->failed) return;
        constexpr bool maximum = operation == DotOperation::initial_norm || operation == DotOperation::residual_norm;
        __shared__ double sums[256], maxima[maximum ? 256 : 1];
        double sum = 0.0, largest = 0.0;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            sum += partial[i];
            if constexpr (maximum) largest = fmax(largest, partial[count + i]);
        }
        sums[threadIdx.x] = sum;
        if constexpr (maximum) maxima[threadIdx.x] = largest;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
                if constexpr (maximum) maxima[threadIdx.x] = fmax(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x) return;
        if (!isfinite(sums[0])) {
            state->failed = 1;
            return;
        }
        if constexpr (operation == DotOperation::update_rz) {
            state->previous_rz = state->rz;
            state->rz          = sums[0];
        } else if constexpr (operation == DotOperation::denominator) {
            state->denominator = sums[0];
            if (sums[0] <= 0.0) state->failed = 1;
        } else {
            if constexpr (operation == DotOperation::initial_norm) state->initial_norm = sums[0];
            state->max_residual = static_cast<float>(maxima[0]);
            bool relative       = sums[0] <= state->initial_norm * relative_tolerance * relative_tolerance;
            state->converged    = maxima[0] <= absolute_tolerance && (relative || maxima[0] <= absolute_tolerance * 0.1f);
        }
    }

    template <bool maximum, bool closed = false>
    __global__ void dot_blocks(Grid grid, const float* a, const float* b, double* partial, const IterationState* state, const double* component_sums, const unsigned char* anchors) {
        if (state->converged || state->failed) return;
        __shared__ double sums[256];
        __shared__ float maxima[maximum ? 256 : 1];
        int i = blockIdx.x * blockDim.x + threadIdx.x, count = grid.layout.count * 512;
        double sum    = i < count ? double(a[i]) * b[i] : 0;
        float largest = 0;
        if constexpr (maximum)
            if (i < count) {
                float h = spacing(grid.layout, i);
                largest = (closed && anchors[i] ? float(fabs(component_sums[i])) : fabsf(a[i])) * grid.layout.h / (h * h * h);
            }
        sums[threadIdx.x] = sum;
        if constexpr (maximum) maxima[threadIdx.x] = largest;
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
                if constexpr (maximum) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (!threadIdx.x) {
            partial[blockIdx.x] = sums[0];
            if constexpr (maximum) partial[gridDim.x + blockIdx.x] = maxima[0];
        }
    }

    template <bool first>
    __global__ void direction_kernel(PressureLevel fine, float* direction, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count) direction[i] = fine.solution[i] + (first ? 0 : float(state->rz / state->previous_rz) * direction[i]);
    }

    template <bool residual, bool adaptive, typename T>
    __global__ void product_kernel(Grid grid, PressureLevel fine, const T* input, float* output, const IterationState* state) {
        if (!residual && (state->converged || state->failed)) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count) output[i] = (residual ? fine.rhs[i] : 0) + (residual ? -1 : 1) * matrix_product<true, adaptive>(grid, fine, input, i);
    }

    __global__ void update_kernel(int count, double* solution, float* residual, const float* direction, const float* product, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        float alpha = float(state->rz / state->denominator);
        solution[i] += (state->rz / state->denominator) * double(direction[i]);
        residual[i] -= alpha * product[i];
    }

    __global__ void count_iteration(IterationState* state) {
        if (!state->converged && !state->failed) ++state->iterations;
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
        return closed != 0;
    }

    int coarse_nodes(PressureLevel fine, PressureLevel coarse, Scratch scratch, int depth, ::cuda::stream_ref stream) {
        int count = coarse.count;
        if (!coarse.nodes) {
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), parent_keys, fine, scratch.keys[0], depth);
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
        ::cuda::launch(stream, ::cuda::distribute<256>(fine.count), link_parents, fine, coarse, scratch.keys[2], depth);
        return count;
    }

    int coarse_edges(PressureLevel fine, PressureLevel coarse, Scratch scratch, ::cuda::stream_ref stream) {
        int count = 0;
        if (fine.edge_count) {
            ::cuda::launch(stream, ::cuda::distribute<256>(fine.edge_count), edge_keys, fine, scratch.keys[0], scratch.values[0]);
            std::size_t bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceRadixSort::SortPairs(scratch.sort_storage, bytes, scratch.keys[0], scratch.keys[1], scratch.values[0], scratch.values[1], fine.edge_count, 0, 64, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceSelect::Unique(scratch.sort_storage, bytes, scratch.keys[1], scratch.keys[2], scratch.count, fine.edge_count, stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            ::cuda::copy_bytes(stream, ::cuda::std::span(scratch.count, 1), ::cuda::std::span(&count, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            unsigned long long last;
            ::cuda::copy_bytes(stream, ::cuda::std::span(scratch.keys[2] + count - 1, 1), ::cuda::std::span(&last, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            if (last == empty_key) --count;
        }
        coarse.edge_count = count;
        ::cuda::fill_bytes(stream, ::cuda::std::span(coarse.adjacency, static_cast<std::size_t>(coarse.count * 24ull)), 255);
        if (count) ::cuda::launch(stream, ::cuda::distribute<256>(count), reduce_edges, coarse, scratch.keys[1], scratch.values[1], scratch.keys[2], fine.edge_count);
        ::cuda::launch(stream, ::cuda::distribute<256>(coarse.count), coarse_diagonal, fine, coarse);
        return count;
    }

    void assemble_rhs(std::span<const PressureLevel> levels, Grid grid, Fields fields, PressureView workspace, ::cuda::stream_ref stream) {
        auto launch = [&]<BoundaryMode boundary_mode>() {
            if (workspace.closed) ::cuda::launch(stream, ::cuda::distribute<256>(levels[0].count), rhs_kernel<boundary_mode, true>, grid, fields, levels[0], workspace.anchors, workspace.solution, workspace.residual);
            else ::cuda::launch(stream, ::cuda::distribute<256>(levels[0].count), rhs_kernel<boundary_mode, false>, grid, fields, levels[0], workspace.anchors, workspace.solution, workspace.residual);
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
            if (l + 1 == workspace.level_count) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), solve_coarsest, level, rhs, workspace.iteration);
            else {
                float upper = l + 1 < grid.layout.levels ? 3.f : 2.f;
                float theta = .5f * (upper + .3f), delta = .5f * (upper - .3f);
                float alpha = 1.f / (theta - delta * delta / (2 * theta)), beta = alpha * delta * delta / (2 * theta);
                auto coarse = levels[l + 1];
                auto launch = [&]<bool fine>() {
                    ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < fine, fine && adaptive, false, true >, grid, level, rhs, level.solution, level.temporary, 1.f / theta, 0, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < fine, fine && adaptive, true, true >, grid, level, rhs, level.temporary, level.solution, alpha, beta, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::distribute<256>(coarse.count), restrict_residual < fine, fine && adaptive >, grid, level, coarse, rhs, workspace.iteration);
                };
                if (l == 0) launch.template operator()<true>();
                else launch.template operator()<false>();
            }
        }
        for (int l = workspace.level_count - 2; l >= 0; --l) {
            auto level       = levels[l];
            const float* rhs = l ? level.rhs : workspace.residual;
            float upper      = l + 1 < grid.layout.levels ? 3.f : 2.f;
            float theta = .5f * (upper + .3f), delta = .5f * (upper - .3f);
            float alpha = 1.f / (theta - delta * delta / (2 * theta)), beta = alpha * delta * delta / (2 * theta);
            ::cuda::launch(stream, ::cuda::distribute<256>(level.count), prolongate, level, levels[l + 1], workspace.iteration);
            auto launch = [&]<bool fine>() {
                ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < fine, fine && adaptive, false, false >, grid, level, rhs, level.solution, level.temporary, 1.f / theta, 0, workspace.iteration);
                ::cuda::launch(stream, ::cuda::distribute<256>(level.count), smooth < fine, fine && adaptive, true, false >, grid, level, rhs, level.temporary, level.solution, alpha, beta, workspace.iteration);
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
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<false>, grid, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::update_rz>, workspace.reductions, count, workspace.iteration, 0, 0);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), direction_kernel<true>, fine, workspace.direction, workspace.iteration);
    }

    template <bool adaptive>
    void iterate_pressure(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        int count = (fine.count + 255) / 256;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), product_kernel<false, adaptive, float>, grid, fine, workspace.direction, workspace.product, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<false>, grid, workspace.direction, workspace.product, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::denominator>, workspace.reductions, count, workspace.iteration, 0, 0);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), update_kernel, fine.count, workspace.solution, workspace.residual, workspace.direction, workspace.product, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<1>()), count_iteration, workspace.iteration);
        component_sums(grid, workspace, stream);
        if (workspace.closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<true, true>, grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, workspace.component_sums, workspace.anchors);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<true>, grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::residual_norm>, workspace.reductions, count, workspace.iteration, config.relative_tolerance, config.divergence_tolerance * .25f);
        precondition<adaptive>(levels, grid, workspace, stream);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), dot_blocks<false>, grid, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, nullptr, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<256>()), finish_dot<DotOperation::update_rz>, workspace.reductions, count, workspace.iteration, 0, 0);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), direction_kernel<false>, fine, workspace.direction, workspace.iteration);
    }

    template <bool adaptive>
    void check_pressure(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        int count = (fine.count + 255) / 256;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<256>()), product_kernel<true, adaptive, double>, grid, fine, workspace.solution, workspace.residual, workspace.iteration);
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
          node_storage(stream, pool),
          edge_storage(stream, pool),
          index_storage(stream, pool),
          anchor_storage(stream, pool),
          field_storage(stream, pool),
          double_storage(stream, pool) {
        view.iteration = iteration_storage.data();
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
        matrix_dirty = true;
        auto cells = std::size_t(layout.capacity) * 512, faces = std::size_t(layout.capacity) * 1728;
        auto edges     = std::size_t(layout.capacity) * (layout.levels > 1 ? 4032 : 1728);
        node_storage   = ::cuda::make_buffer<Node>(stream, pool, cells, ::cuda::no_init);
        edge_storage   = ::cuda::make_buffer<Edge>(stream, pool, edges, ::cuda::no_init);
        index_storage  = ::cuda::make_buffer<int>(stream, pool, cells * 3, ::cuda::no_init);
        anchor_storage = ::cuda::make_buffer<unsigned char>(stream, pool, cells, ::cuda::no_init);
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
        auto levels = std::span(hierarchy);
        iteration_graph.reset();
        auto& fine      = hierarchy[0];
        fine.count      = grid.layout.count * 512;
        fine.edge_count = grid.layout.count * (grid.layout.levels > 1 ? 4032 : 1728);
        if (grid.layout.levels > 1) view.closed = fine_matrix<true>(levels, grid, view, stream);
        else view.closed = fine_matrix<false>(levels, grid, view, stream);
        view.level_count = 1;
        for (int depth = 1; hierarchy[depth - 1].count > 128; ++depth) {
            auto& previous = hierarchy[depth - 1];
            auto& level    = hierarchy[depth];
            int count      = coarse_nodes(previous, {}, scratch, depth, stream);
            if (count > level.capacity) {
                int capacity  = (count + 63) / 64 * 64;
                auto& nodes   = coarse_node_storage[depth - 1];
                auto& edges   = coarse_edge_storage[depth - 1];
                auto& indices = coarse_index_storage[depth - 1];
                auto& fields  = coarse_field_storage[depth - 1];
                nodes         = ::cuda::make_buffer<Node>(stream, pool, capacity, ::cuda::no_init);
                edges         = ::cuda::make_buffer<Edge>(stream, pool, capacity * 12ull, ::cuda::no_init);
                indices       = ::cuda::make_buffer<int>(stream, pool, capacity * 33ull, ::cuda::no_init);
                fields        = ::cuda::make_buffer<float>(stream, pool, capacity * 5ull, ::cuda::no_init);
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
            }
            level.count = count;
            coarse_nodes(previous, level, scratch, depth, stream);
            level.edge_count = coarse_edges(previous, level, scratch, stream);
            ++view.level_count;
        }
        matrix_dirty = false;
    }

    int Pressure::project(Grid grid, Fields fields, Scratch scratch) {
        if (matrix_dirty) build(grid, scratch);
        auto levels = std::span(hierarchy).first(view.level_count);
        assemble_rhs(levels, grid, fields, view, stream);
        ::cuda::fill_bytes(stream, ::cuda::std::span(view.iteration, 1), 0);
        if (grid.layout.levels > 1) begin_pressure<true>(levels, grid, view, config, stream);
        else begin_pressure<false>(levels, grid, view, config, stream);
        if (!iteration_graph) {
            cudaGraph_t captured{};
            if (auto cuda_result = cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            for (int i = 0; i < 8; ++i)
                if (grid.layout.levels > 1) iterate_pressure<true>(levels, grid, view, config, stream);
                else iterate_pressure<false>(levels, grid, view, config, stream);
            if (auto cuda_result = cudaStreamEndCapture(stream.get(), &captured); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph{captured, cudaGraphDestroy};
            cudaGraphExec_t executable{};
            if (auto cuda_result = cudaGraphInstantiate(&executable, graph.get(), 0); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            iteration_graph.reset(executable);
        }
        IterationState status{};
        for (;;) {
            int remaining = config.max_iterations - status.iterations;
            if (remaining < 8) {
                for (int i = 0; i < remaining; ++i)
                    if (grid.layout.levels > 1) iterate_pressure<true>(levels, grid, view, config, stream);
                    else iterate_pressure<false>(levels, grid, view, config, stream);
            } else if (auto cuda_result = cudaGraphLaunch(iteration_graph.get(), stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
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
        return status.iterations;
    }

}
