#include "cuda.cuh"
#include <cub/cub.cuh>

namespace pyro::cuda::sparse {
    __global__ void initialize_fine(Grid grid, PressureLevel fine, int* parents, int* open, unsigned char* anchors) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        parents[i]    = i;
        open[i]       = 0;
        anchors[i]    = 0;
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
    __device__ int interface_edge(Grid grid, int group, int a, int b) {
        int low = min(a, b), high = max(a, b);
        return grid.layout.count * 1728 + group * 6 + low * 3 - low * (low - 1) / 2 + high - low - 1;
    }
    __device__ Edge physical_edge(Grid grid, int f) {
        int count = grid.layout.count * 1728;
        if (f < count) {
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
    __global__ void fine_edges(Grid grid, PressureLevel fine, const unsigned char* anchors) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= fine.edge_count) return;
        Edge edge = physical_edge(grid, f);
        if (edge.left < 0 || edge.right < 0 || anchors[edge.left] || anchors[edge.right]) edge.weight = 0;
        fine.edges[f] = edge;
        if (f < grid.layout.count * 1728) fine.face_weights[f] = edge.weight;
    }
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
                        if (j < 0 || anchors[j]) boundary += face.weight;
                        if (face.group >= 0 && spacing(grid.layout, i) < spacing(grid.layout, j)) {
                            Links group = grid.links[face.group];
                            int ordinal = 0;
                            while (group.face[ordinal] != f) ++ordinal;
                            for (int q = 0; q < 4; ++q)
                                if (q != ordinal) {
                                    Edge edge    = physical_edge(grid, interface_edge(grid, face.group, ordinal, q));
                                    int neighbor = edge.left == i ? edge.right : edge.left;
                                    diagonal += edge.weight;
                                    if (anchors[neighbor]) boundary += edge.weight;
                                }
                        }
                    }
        fine.boundary[i] = anchors[i] ? 1 : grid.fluid[i] ? float(boundary) : 0;
        fine.diagonal[i] = anchors[i] ? 1 : grid.fluid[i] ? float(diagonal) : 0;
    }
    bool fine_matrix(Grid grid, PressureWorkspace workspace, bool obstacles, cudaStream_t stream) {
        if (auto cuda_result = cudaMemsetAsync(workspace.iteration, 0, sizeof(IterationState), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        auto fine = workspace.levels[0];
        initialize_fine<<<(fine.count + 255) / 256, 256, 0, stream>>>(grid, fine, workspace.components, workspace.open, workspace.anchors);
        if (obstacles) {
            connect_components<<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, workspace.components);
            mark_open<<<(fine.count + 255) / 256, 256, 0, stream>>>(grid, workspace.components, workspace.open);
            mark_anchors<<<(fine.count + 255) / 256, 256, 0, stream>>>(grid, workspace.components, workspace.open, workspace.anchors, workspace.iteration);
        }
        fine_edges<<<(fine.edge_count + 255) / 256, 256, 0, stream>>>(grid, fine, workspace.anchors);
        fine_diagonal<<<(fine.count + 255) / 256, 256, 0, stream>>>(grid, fine, workspace.anchors);
        int closed = 0;
        if (obstacles) {
            if (auto cuda_result = cudaMemcpyAsync(&closed, &workspace.iteration->iterations, sizeof(int), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        }
        return closed != 0;
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
    int coarse_nodes(PressureLevel fine, PressureLevel coarse, Scratch scratch, int depth, cudaStream_t stream) {
        int count = coarse.count;
        if (!coarse.nodes) {
            parent_keys<<<(fine.count + 255) / 256, 256, 0, stream>>>(fine, scratch.keys[0], depth);
            std::size_t bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceRadixSort::SortKeys(scratch.sort_storage, bytes, scratch.keys[0], scratch.keys[1], fine.count, 0, 64, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceSelect::Unique(scratch.sort_storage, bytes, scratch.keys[1], scratch.keys[2], scratch.count, fine.count, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMemcpyAsync(&count, scratch.count, sizeof(int), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            return count;
        }
        coarse.count = count;
        if (auto cuda_result = cudaMemsetAsync(coarse.children, 255, count * 8ull * sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        unpack_nodes<<<(count + 255) / 256, 256, 0, stream>>>(coarse, scratch.keys[2]);
        link_parents<<<(fine.count + 255) / 256, 256, 0, stream>>>(fine, coarse, scratch.keys[2], depth);
        return count;
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
    int coarse_edges(PressureLevel fine, PressureLevel coarse, Scratch scratch, cudaStream_t stream) {
        int count = 0;
        if (fine.edge_count) {
            edge_keys<<<(fine.edge_count + 255) / 256, 256, 0, stream>>>(fine, scratch.keys[0], scratch.values[0]);
            std::size_t bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceRadixSort::SortPairs(scratch.sort_storage, bytes, scratch.keys[0], scratch.keys[1], scratch.values[0], scratch.values[1], fine.edge_count, 0, 64, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            bytes = scratch.sort_bytes;
            if (auto cuda_result = cub::DeviceSelect::Unique(scratch.sort_storage, bytes, scratch.keys[1], scratch.keys[2], scratch.count, fine.edge_count, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaMemcpyAsync(&count, scratch.count, sizeof(int), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            unsigned long long last;
            if (auto cuda_result = cudaMemcpyAsync(&last, scratch.keys[2] + count - 1, sizeof(last), cudaMemcpyDeviceToHost, stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (auto cuda_result = cudaStreamSynchronize(stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            if (last == empty_key) --count;
        }
        coarse.edge_count = count;
        if (auto cuda_result = cudaMemsetAsync(coarse.adjacency, 255, coarse.count * 24ull * sizeof(int), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        if (count) reduce_edges<<<(count + 255) / 256, 256, 0, stream>>>(coarse, scratch.keys[1], scratch.values[1], scratch.keys[2], fine.edge_count);
        coarse_diagonal<<<(coarse.count + 255) / 256, 256, 0, stream>>>(fine, coarse);
        return count;
    }
    template <typename T>
    __device__ T matrix_product(Grid grid, PressureLevel level, const T* x, int i, bool fine) {
        if (level.diagonal[i] <= 0) return 0;
        T result = level.boundary[i] * x[i];
        if (fine) {
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
    __global__ void rhs_kernel(Grid grid, Fields fields, PressureLevel fine, const unsigned char* anchors, double* solution, float* residual) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= fine.count) return;
        float h     = spacing(grid.layout, i);
        double flux = -double(grid.solid_divergence[i]) * h * h * h;
        for (int a = 0; a < 3; ++a)
            for (int sign = -1; sign <= 1; sign += 2)
                for (int f : cell_faces(grid, i, a, sign).face)
                    if (f >= 0) flux += sign * (double(grid.faces[f].area) * fields.velocity[f] + grid.faces[f].solid_flux);
        fine.rhs[i] = residual[i] = grid.fluid[i] && !anchors[i] ? -flux / grid.layout.h : 0;
        solution[i]               = 0;
    }
    void assemble_rhs(Grid grid, Fields fields, PressureWorkspace workspace, cudaStream_t stream) {
        rhs_kernel<<<(workspace.levels[0].count + 255) / 256, 256, 0, stream>>>(grid, fields, workspace.levels[0], workspace.anchors, workspace.solution, workspace.residual);
    }
    __global__ void smooth(Grid grid, PressureLevel level, const float* rhs, const float* input, float* output, bool fine, bool second, float upper, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= level.count) return;
        float theta = .5f * (upper + .3f), delta = .5f * (upper - .3f);
        float alpha    = 1.f / (theta - delta * delta / (2 * theta));
        float beta     = alpha * delta * delta / (2 * theta);
        float residual = level.diagonal[i] > 0 ? (rhs[i] - matrix_product(grid, level, input, i, fine)) / level.diagonal[i] : 0;
        output[i]      = level.diagonal[i] > 0 ? input[i] + (second ? alpha : 1.f / theta) * residual + (second ? beta * (input[i] - output[i]) : 0) : 0;
    }
    __global__ void restrict_residual(Grid grid, PressureLevel fine, PressureLevel coarse, const float* rhs, bool finest, const IterationState* state) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= coarse.count) return;
        double value = 0;
        for (int c = 0; c < 8; ++c) {
            int j = coarse.children[i * 8 + c];
            if (j >= 0) value += rhs[j] - matrix_product(grid, fine, fine.solution, j, finest);
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
    void precondition(Grid grid, PressureWorkspace workspace, cudaStream_t stream) {
        for (int l = 0; l < workspace.level_count; ++l) {
            auto level       = workspace.levels[l];
            const float* rhs = l ? level.rhs : workspace.residual;
            if (l + 1 == workspace.level_count) solve_coarsest<<<1, 256, 0, stream>>>(level, rhs, workspace.iteration);
            else {
                if (auto cuda_result = cudaMemsetAsync(level.solution, 0, level.count * sizeof(float), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
                smooth<<<(level.count + 255) / 256, 256, 0, stream>>>(grid, level, rhs, level.solution, level.temporary, l == 0, false, l + 1 < grid.layout.levels ? 3.f : 2.f, workspace.iteration);
                smooth<<<(level.count + 255) / 256, 256, 0, stream>>>(grid, level, rhs, level.temporary, level.solution, l == 0, true, l + 1 < grid.layout.levels ? 3.f : 2.f, workspace.iteration);
                auto coarse = workspace.levels[l + 1];
                restrict_residual<<<(coarse.count + 255) / 256, 256, 0, stream>>>(grid, level, coarse, rhs, l == 0, workspace.iteration);
            }
        }
        for (int l = workspace.level_count - 2; l >= 0; --l) {
            auto level       = workspace.levels[l];
            const float* rhs = l ? level.rhs : workspace.residual;
            prolongate<<<(level.count + 255) / 256, 256, 0, stream>>>(level, workspace.levels[l + 1], workspace.iteration);
            smooth<<<(level.count + 255) / 256, 256, 0, stream>>>(grid, level, rhs, level.solution, level.temporary, l == 0, false, l + 1 < grid.layout.levels ? 3.f : 2.f, workspace.iteration);
            smooth<<<(level.count + 255) / 256, 256, 0, stream>>>(grid, level, rhs, level.temporary, level.solution, l == 0, true, l + 1 < grid.layout.levels ? 3.f : 2.f, workspace.iteration);
        }
    }
    __global__ void sum_closed_residual(int count, const float* residual, const int* components, const int* open, double* sums) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count && !open[components[i]]) atomicAdd(sums + components[i], double(residual[i]));
    }
    void component_sums(Grid grid, PressureWorkspace workspace, cudaStream_t stream) {
        if (!workspace.closed) return;
        int count = grid.layout.count * 512;
        if (auto cuda_result = cudaMemsetAsync(workspace.component_sums, 0, count * sizeof(double), stream); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
        sum_closed_residual<<<(count + 255) / 256, 256, 0, stream>>>(count, workspace.residual, workspace.components, workspace.open, workspace.component_sums);
    }
    __global__ void finish_dot(double* partial, int count, IterationState* state, int operation, float relative_tolerance, float absolute_tolerance) {
        if (state->converged || state->failed) return;
        __shared__ double sums[256], maxima[256];
        double sum = 0.0, largest = 0.0;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            sum += partial[i];
            largest = fmax(largest, partial[count + i]);
        }
        sums[threadIdx.x]   = sum;
        maxima[threadIdx.x] = largest;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
                maxima[threadIdx.x] = fmax(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x) return;
        if (!isfinite(sums[0])) {
            state->failed = 1;
            return;
        }
        if (operation == 0) {
            state->previous_rz = state->rz;
            state->rz          = sums[0];
        } else if (operation == 1) {
            state->denominator = sums[0];
            if (sums[0] <= 0.0) state->failed = 1;
        } else {
            if (operation == 2) state->initial_norm = sums[0];
            state->max_residual = static_cast<float>(maxima[0]);
            bool relative       = sums[0] <= state->initial_norm * relative_tolerance * relative_tolerance;
            state->converged    = maxima[0] <= absolute_tolerance && (relative || maxima[0] <= absolute_tolerance * 0.1f);
        }
    }

    __global__ void dot_blocks(Grid grid, const float* a, const float* b, double* partial, const IterationState* state, bool maximum, const double* component_sums = nullptr, const unsigned char* anchors = nullptr) {
        if (state->converged || state->failed) return;
        __shared__ double sums[256];
        __shared__ float maxima[256];
        int i = blockIdx.x * blockDim.x + threadIdx.x, count = grid.layout.count * 512;
        double sum    = i < count ? double(a[i]) * b[i] : 0;
        float largest = 0;
        if (i < count && maximum) {
            float h = spacing(grid.layout, i);
            largest = (anchors && anchors[i] ? float(fabs(component_sums[i])) : fabsf(a[i])) * grid.layout.h / (h * h * h);
        }
        sums[threadIdx.x]   = sum;
        maxima[threadIdx.x] = largest;
        __syncthreads();
        for (int stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
                maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (!threadIdx.x) {
            partial[blockIdx.x]             = sums[0];
            partial[gridDim.x + blockIdx.x] = maxima[0];
        }
    }
    __global__ void direction_kernel(PressureLevel fine, float* direction, const IterationState* state, bool first) {
        if (state->converged || state->failed) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count) direction[i] = fine.solution[i] + (first ? 0 : float(state->rz / state->previous_rz) * direction[i]);
    }
    template <typename T>
    __global__ void product_kernel(Grid grid, PressureLevel fine, const T* input, float* output, const IterationState* state, bool residual) {
        if (!residual && (state->converged || state->failed)) return;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < fine.count) output[i] = (residual ? fine.rhs[i] : 0) + (residual ? -1 : 1) * matrix_product(grid, fine, input, i, true);
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
    void begin_pressure(Grid grid, PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        int count = (fine.count + 255) / 256;
        component_sums(grid, workspace, stream);
        dot_blocks<<<count, 256, 0, stream>>>(grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, true, workspace.closed ? workspace.component_sums : nullptr, workspace.closed ? workspace.anchors : nullptr);
        finish_dot<<<1, 256, 0, stream>>>(workspace.reductions, count, workspace.iteration, 2, config.pressure_relative_tolerance, config.divergence_tolerance * .25f);
        precondition(grid, workspace, stream);
        dot_blocks<<<count, 256, 0, stream>>>(grid, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, false);
        finish_dot<<<1, 256, 0, stream>>>(workspace.reductions, count, workspace.iteration, 0, 0, 0);
        direction_kernel<<<count, 256, 0, stream>>>(fine, workspace.direction, workspace.iteration, true);
    }
    void iterate_pressure(Grid grid, PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        int count = (fine.count + 255) / 256;
        product_kernel<<<count, 256, 0, stream>>>(grid, fine, workspace.direction, workspace.product, workspace.iteration, false);
        dot_blocks<<<count, 256, 0, stream>>>(grid, workspace.direction, workspace.product, workspace.reductions, workspace.iteration, false);
        finish_dot<<<1, 256, 0, stream>>>(workspace.reductions, count, workspace.iteration, 1, 0, 0);
        update_kernel<<<count, 256, 0, stream>>>(fine.count, workspace.solution, workspace.residual, workspace.direction, workspace.product, workspace.iteration);
        count_iteration<<<1, 1, 0, stream>>>(workspace.iteration);
        component_sums(grid, workspace, stream);
        dot_blocks<<<count, 256, 0, stream>>>(grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, true, workspace.closed ? workspace.component_sums : nullptr, workspace.closed ? workspace.anchors : nullptr);
        finish_dot<<<1, 256, 0, stream>>>(workspace.reductions, count, workspace.iteration, 3, config.pressure_relative_tolerance, config.divergence_tolerance * .25f);
        precondition(grid, workspace, stream);
        dot_blocks<<<count, 256, 0, stream>>>(grid, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, false);
        finish_dot<<<1, 256, 0, stream>>>(workspace.reductions, count, workspace.iteration, 0, 0, 0);
        direction_kernel<<<count, 256, 0, stream>>>(fine, workspace.direction, workspace.iteration, false);
    }
    void check_pressure(Grid grid, PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        int count = (fine.count + 255) / 256;
        product_kernel<<<count, 256, 0, stream>>>(grid, fine, workspace.solution, workspace.residual, workspace.iteration, true);
        component_sums(grid, workspace, stream);
        dot_blocks<<<count, 256, 0, stream>>>(grid, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, true, workspace.closed ? workspace.component_sums : nullptr, workspace.closed ? workspace.anchors : nullptr);
        finish_dot<<<1, 256, 0, stream>>>(workspace.reductions, count, workspace.iteration, 3, config.pressure_relative_tolerance, config.divergence_tolerance * .5f);
    }
    __global__ void project_faces(Grid grid, Fields fields, const double* pressure) {
        int f = blockIdx.x * blockDim.x + threadIdx.x;
        if (f >= grid.layout.count * 1728) return;
        Face face = grid.faces[f];
        if (face.left < 0 && face.right < 0) return;
        if (face.weight > 0) {
            double difference = (face.left >= 0 ? pressure[face.left] : 0) - (face.right >= 0 ? pressure[face.right] : 0);
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
            float h            = ldexpf(grid.layout.h, grid.layout.blocks[f / 1728].level);
            fields.velocity[f] = face.solid_flux / (h * h);
        }
    }
    void apply_pressure(Grid grid, Fields fields, const double* pressure, cudaStream_t stream) {
        project_faces<<<(grid.layout.count * 1728 + 255) / 256, 256, 0, stream>>>(grid, fields, pressure);
    }
} // namespace pyro::cuda::sparse
