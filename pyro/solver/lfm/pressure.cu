#include "pressure.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include <cub/block/block_reduce.cuh>
#include <cuda/algorithm>
#include <cuda/functional>
#include <cuda/launch>
#include <cuda/std/span>

namespace pyro::lfm {
    enum struct DotOperation { update_rz, denominator, initial_norm, residual_norm };

    template <bool regular>
    __device__ void product_block(PressureLevel fine, const float* direction, float* result, double* partial) {
        __shared__ float x[1000];
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        int3 origin = coordinate(fine.layout, blockIdx.x * tile_cells);
        for (int j = threadIdx.x; j < 1000; j += blockDim.x) {
            int3 p{origin.x + j % 10 - 1, origin.y + j / 10 % 10 - 1, origin.z + j / 100 - 1};
            x[j] = regular ? direction[index(fine.layout, p)] : read(fine.layout, direction, p);
        }
        __syncthreads();
        double sum = 0;
        for (int j = threadIdx.x; j < tile_cells; j += blockDim.x) {
            int i = blockIdx.x * tile_cells + j;
            int s = (j & 7) + 1 + 10 * (((j >> 3) & 7) + 1) + 100 * ((j >> 6) + 1);
            int3 p{origin.x + (j & 7), origin.y + ((j >> 3) & 7), origin.z + (j >> 6)};
            float value = 0;
            if (regular || fine.diagonal[i] > 0) {
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) {
                    int stride = axis == 0 ? 1 : axis == 1 ? 10 : 100;
                    float a    = regular ? fine.regular_weight : fine.weight[axis][i];
                    float b    = regular ? fine.regular_weight : read(fine.layout, fine.weight[axis], int3{p.x + (axis == 0 ? 1 : 0), p.y + (axis == 1 ? 1 : 0), p.z + (axis == 2 ? 1 : 0)});
                    value += a * (x[s] - x[s - stride]) + b * (x[s] - x[s + stride]);
                }
            }
            result[i] = value;
            sum += static_cast<double>(direction[i]) * value;
        }
        double value = cub::BlockReduce<double, block_threads>(reduction).Sum(sum);
        if (!threadIdx.x) partial[blockIdx.x] = value;
    }

    __global__ void fine_matrix(Grid grid, PressureLevel level, int* parents) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i          = blockIdx.x * tile_cells + local;
            int3 p         = coordinate(level.layout, i);
            parents[i]     = i;
            float diagonal = 0;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                bool negative         = grid.fluid[i] && !grid.boundary[axis][index(grid.faces[axis], p)];
                level.weight[axis][i] = negative ? 1 : 0;
                if (grid.fluid[i]) diagonal += negative + !grid.boundary[axis][index(grid.faces[axis], int3{p.x + (axis == 0 ? 1 : 0), p.y + (axis == 1 ? 1 : 0), p.z + (axis == 2 ? 1 : 0)})];
            }
            level.diagonal[i] = diagonal;
        }
    }

    __global__ void connect_components(PressureLevel level, int* parents) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i  = blockIdx.x * tile_cells + local;
            int3 p = coordinate(level.layout, i);
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                if (!level.weight[axis][i]) continue;
                int j = index(level.layout, int3{p.x + (axis == 0 ? -1 : 0), p.y + (axis == 1 ? -1 : 0), p.z + (axis == 2 ? -1 : 0)});
                int a = i, b = j;
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
        }
    }

    __global__ void label_components(Grid grid, PressureView workspace) {
        __shared__ typename cub::BlockReduce<int, block_threads>::TempStorage reduction;
        int lower = 0x7fffffff, upper = -1;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i    = blockIdx.x * tile_cells + local;
            int root = -1;
            if (grid.fluid[i]) {
                root       = i;
                int parent = atomicAdd(workspace.parents + root, 0);
                while (parent != root) {
                    int next = atomicAdd(workspace.parents + parent, 0);
                    atomicMin(workspace.parents + root, next);
                    root   = parent;
                    parent = next;
                }
            }
            atomicExch(workspace.parents + i, root);
            if (root >= 0) lower = min(lower, root);
            upper = max(upper, root);
        }
        int lo = cub::BlockReduce<int, block_threads>(reduction).Reduce(lower, ::cuda::minimum<>{});
        if (!threadIdx.x) workspace.block_components[blockIdx.x] = lo;
        __syncthreads();
        int hi = cub::BlockReduce<int, block_threads>(reduction).Reduce(upper, ::cuda::maximum<>{});
        if (!threadIdx.x) workspace.block_components[blockIdx.x] = hi < 0 ? -1 : hi == workspace.block_components[blockIdx.x] ? hi : -2;
    }

    template <bool count>
    __global__ void component_sums(PressureView workspace, const IterationState* state) {
        if constexpr (!count)
            if (state->converged || state->failed) return;
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        int root = workspace.block_components[blockIdx.x];
        if (root == -1) return;
        double sum = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local, component = workspace.parents[i];
            if (component < 0) continue;
            double value = count ? 1 : workspace.residual[i];
            if (root >= 0) sum += value;
            else {
                if constexpr (count) atomicAdd(workspace.component_counts + component, 1);
                else atomicAdd(workspace.component_sums + component, value);
            }
        }
        if (root >= 0) {
            double total = cub::BlockReduce<double, block_threads>(reduction).Sum(sum);
            if (!threadIdx.x) {
                if constexpr (count) atomicAdd(workspace.component_counts + root, static_cast<int>(total));
                else atomicAdd(workspace.component_sums + root, total);
            }
        }
    }

    __global__ void center_residual(PressureLevel level, PressureView workspace) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        double sum = 0, largest = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local, root = workspace.parents[i];
            float r               = root >= 0 ? static_cast<float>(workspace.residual[i] - workspace.component_sums[root] / workspace.component_counts[root]) : 0;
            workspace.residual[i] = r;
            sum += static_cast<double>(r) * r;
            largest = fmax(largest, fabs(static_cast<double>(r)));
        }
        double value = cub::BlockReduce<double, block_threads>(reduction).Sum(sum);
        if (!threadIdx.x) workspace.reductions[blockIdx.x] = value;
        __syncthreads();
        value = cub::BlockReduce<double, block_threads>(reduction).Reduce(largest, ::cuda::maximum<>{});
        if (!threadIdx.x) workspace.reductions[level.layout.blocks + blockIdx.x] = value;
    }

    __global__ void clear_component_sums(PressureView workspace, int blocks) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        int block = blockIdx.x * blockDim.x + threadIdx.x;
        if (block >= blocks) return;
        int root = workspace.block_components[block];
        if (root >= 0) atomicExch(reinterpret_cast<unsigned long long*>(workspace.component_sums + root), 0ull);
        else if (root == -2)
            for (int j = 0; j < tile_cells; ++j) {
                int component = workspace.parents[block * tile_cells + j];
                if (component >= 0) atomicExch(reinterpret_cast<unsigned long long*>(workspace.component_sums + component), 0ull);
            }
    }

    __global__ void coarsen_matrix(PressureLevel fine, PressureLevel coarse) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i  = blockIdx.x * tile_cells + local;
            int3 p = coordinate(coarse.layout, i);
            float weights[3]{}, diagonal = 0;
            if (contains(coarse.layout, p)) {
                for (int z = 0; z < 2; ++z)
                    for (int y = 0; y < 2; ++y)
                        for (int x = 0; x < 2; ++x) {
                            int3 q{2 * p.x + x, 2 * p.y + y, 2 * p.z + z};
                            if (!contains(fine.layout, q)) continue;
                            int j = index(fine.layout, q);
                            int parity[3]{x, y, z};
#pragma unroll
                            for (int axis = 0; axis < 3; ++axis) {
                                if (!parity[axis]) {
                                    weights[axis] += fine.weight[axis][j];
                                    diagonal += fine.weight[axis][j];
                                } else diagonal += read(fine.layout, fine.weight[axis], int3{q.x + (axis == 0 ? 1 : 0), q.y + (axis == 1 ? 1 : 0), q.z + (axis == 2 ? 1 : 0)});
                            }
                        }
            }
            coarse.diagonal[i] = diagonal;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) coarse.weight[axis][i] = weights[axis];
        }
    }

    __global__ void classify(PressureLevel level) {
        __shared__ typename cub::BlockReduce<int, block_threads>::TempStorage reduction;
        int3 origin = coordinate(level.layout, blockIdx.x * tile_cells);
        int active = 0, irregular = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) active |= level.diagonal[blockIdx.x * tile_cells + local] > 0;
        for (int j = threadIdx.x; j < 1728; j += blockDim.x) {
            int3 p{origin.x + j % 12 - 2, origin.y + j / 12 % 12 - 2, origin.z + j / 144 - 2};
            if (!contains(level.layout, p)) irregular = 1;
            else {
                int i = index(level.layout, p);
                irregular |= level.diagonal[i] != 6 * level.regular_weight;
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) irregular |= level.weight[axis][i] != level.regular_weight;
            }
        }
        int any = cub::BlockReduce<int, block_threads>(reduction).Reduce(active, ::cuda::maximum<>{});
        if (!threadIdx.x) level.kind[blockIdx.x] = any ? 1 : 0;
        __syncthreads();
        int general = cub::BlockReduce<int, block_threads>(reduction).Reduce(irregular, ::cuda::maximum<>{});
        if (!threadIdx.x && level.kind[blockIdx.x] && general) level.kind[blockIdx.x] = 2;
    }

    template <bool post, bool regular, bool dot = false>
    __global__ void smooth_transfer(PressureLevel fine, PressureLevel coarse, const float* rhs, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        if (fine.kind[blockIdx.x] != (regular ? 1 : 2)) return;
        __shared__ float x[1728], b[1728];
        __shared__ float diagonal[regular ? 1 : 1728], weight[regular ? 1 : 3][regular ? 1 : 1728];
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        int3 origin = coordinate(fine.layout, blockIdx.x * tile_cells);
        for (int j = threadIdx.x; j < 1728; j += blockDim.x) {
            int3 p{origin.x + j % 12 - 2, origin.y + j / 12 % 12 - 2, origin.z + j / 144 - 2};
            bool valid = regular || contains(fine.layout, p);
            int i      = valid ? index(fine.layout, p) : 0;
            float d    = regular ? 6 * fine.regular_weight : valid ? fine.diagonal[i] : 0;
            b[j]       = valid ? rhs[i] : 0;
            if constexpr (!regular) {
                diagonal[j] = d;
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) weight[axis][j] = valid ? fine.weight[axis][i] : 0;
            }
            if constexpr (post) x[j] = regular || d > 0 ? fine.solution[i] + 2 * coarse.temporary[index(coarse.layout, {p.x / 2, p.y / 2, p.z / 2})] : 0;
            else x[j] = (regular || d > 0) && !((p.x + p.y + p.z) & 1) ? b[j] / d : 0;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < 1000; j += blockDim.x) {
            int px = j % 10 + 1, py = j / 10 % 10 + 1, pz = j / 100 + 1;
            int s = px + 12 * py + 144 * pz;
            if (!((px + py + pz) & 1)) continue;
            float d     = regular ? 6 * fine.regular_weight : diagonal[s];
            float value = b[s];
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                int stride = axis == 0 ? 1 : axis == 1 ? 12 : 144;
                float a    = regular ? fine.regular_weight : weight[axis][s];
                float c    = regular ? fine.regular_weight : weight[axis][s + stride];
                value += a * x[s - stride] + c * x[s + stride];
            }
            x[s] = regular || d > 0 ? value / d : 0;
        }
        __syncthreads();
        double sum = 0;
        for (int j = threadIdx.x; j < tile_cells; j += blockDim.x) {
            int px = (j & 7) + 2, py = ((j >> 3) & 7) + 2, pz = (j >> 6) + 2;
            int s = px + 12 * py + 144 * pz, i = blockIdx.x * tile_cells + j;
            float d         = regular ? 6 * fine.regular_weight : diagonal[s];
            float neighbors = 0;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                int stride = axis == 0 ? 1 : axis == 1 ? 12 : 144;
                float a    = regular ? fine.regular_weight : weight[axis][s];
                float c    = regular ? fine.regular_weight : weight[axis][s + stride];
                neighbors += a * x[s - stride] + c * x[s + stride];
            }
            if constexpr (post) {
                float result      = !((px + py + pz) & 1) && (regular || d > 0) ? (b[s] + neighbors) / d : x[s];
                fine.temporary[i] = result;
                if constexpr (dot) sum += static_cast<double>(result) * b[s];
            } else {
                fine.solution[i] = x[s];
                b[s]             = regular || d > 0 ? b[s] - (d * x[s] - neighbors) : 0;
            }
        }
        __syncthreads();
        if constexpr (post) {
            if constexpr (dot) {
                double value = cub::BlockReduce<double, block_threads>(reduction).Sum(sum);
                if (!threadIdx.x) partial[blockIdx.x] = value;
            }
        } else {
            for (int j = threadIdx.x; j < 64; j += blockDim.x) {
                int px = j & 3, py = (j >> 2) & 3, pz = j >> 4;
                float value = 0;
                for (int z = 0; z < 2; ++z)
                    for (int y = 0; y < 2; ++y)
                        for (int a = 0; a < 2; ++a) value += b[2 + 2 * px + a + 12 * (2 + 2 * py + y) + 144 * (2 + 2 * pz + z)];
                int3 p{origin.x / 2 + px, origin.y / 2 + py, origin.z / 2 + pz};
                if (contains(coarse.layout, p)) coarse.rhs[index(coarse.layout, p)] = value;
            }
        }
    }

    template <bool dot>
    __global__ void empty_transfer(PressureLevel fine, PressureLevel coarse, double* partial, const IterationState* state) {
        if (state->converged || state->failed || fine.kind[blockIdx.x]) return;
        int3 origin = coordinate(fine.layout, blockIdx.x * tile_cells);
        for (int j = threadIdx.x; j < tile_cells; j += blockDim.x) {
            int i            = blockIdx.x * tile_cells + j;
            fine.solution[i] = fine.temporary[i] = 0;
        }
        for (int j = threadIdx.x; j < 64; j += blockDim.x) {
            int3 p{origin.x / 2 + (j & 3), origin.y / 2 + ((j >> 2) & 3), origin.z / 2 + (j >> 4)};
            if (contains(coarse.layout, p)) coarse.rhs[index(coarse.layout, p)] = 0;
        }
        if constexpr (dot)
            if (!threadIdx.x) partial[blockIdx.x] = 0;
    }

    __global__ void coarsest(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ float x[512], b[512], d[512], w[3][512];
        int nx = level.layout.size.x, ny = level.layout.size.y, nz = level.layout.size.z;
        int count = nx * ny * nz;
        for (int j = threadIdx.x; j < count; j += blockDim.x) {
            int3 p{j % nx, j / nx % ny, j / (nx * ny)};
            int i = index(level.layout, p);
            x[j]  = 0;
            b[j]  = rhs[i];
            d[j]  = level.diagonal[i];
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) w[axis][j] = level.weight[axis][i];
        }
        __syncthreads();
        for (int sweep = 0; sweep < 12; ++sweep)
            for (int phase = 0; phase < 4; ++phase) {
                int color = phase == 0 || phase == 3 ? 0 : 1;
                for (int j = threadIdx.x; j < count; j += blockDim.x) {
                    int p[3]{j % nx, j / nx % ny, j / (nx * ny)}, size[3]{nx, ny, nz};
                    if (((p[0] + p[1] + p[2]) & 1) != color || d[j] <= 0) continue;
                    float value = b[j];
#pragma unroll
                    for (int axis = 0; axis < 3; ++axis) {
                        int stride = axis == 0 ? 1 : axis == 1 ? nx : nx * ny;
                        if (p[axis] > 0) value += w[axis][j] * x[j - stride];
                        if (p[axis] + 1 < size[axis]) value += w[axis][j + stride] * x[j + stride];
                    }
                    x[j] = value / d[j];
                }
                __syncthreads();
            }
        for (int j = threadIdx.x; j < tile_cells; j += blockDim.x) level.temporary[j] = 0;
        __syncthreads();
        for (int j = threadIdx.x; j < count; j += blockDim.x) level.temporary[index(level.layout, {j % nx, j / nx % ny, j / (nx * ny)})] = x[j];
    }

    __global__ void dot_kernel(PressureLevel level, const float* a, const float* b, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        double sum = 0;
        for (int j = threadIdx.x; j < tile_cells; j += blockDim.x) {
            int i = blockIdx.x * tile_cells + j;
            sum += static_cast<double>(a[i]) * b[i];
        }
        double value = cub::BlockReduce<double, block_threads>(reduction).Sum(sum);
        if (!threadIdx.x) partial[blockIdx.x] = value;
    }

    template <DotOperation operation>
    __global__ void finish_dot(double* partial, int count, IterationState* state, float relative_tolerance, float absolute_tolerance) {
        if (state->converged || state->failed) return;
        constexpr bool maximum = operation == DotOperation::initial_norm || operation == DotOperation::residual_norm;
        __shared__ double sums[block_threads], maxima[maximum ? block_threads : 1];
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

    __global__ void rhs_kernel(Grid grid, Velocity velocity, PressureView workspace) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i       = blockIdx.x * tile_cells + local;
            int3 p      = coordinate(grid.cells, i);
            double flux = 0;
            if (grid.fluid[i])
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) flux += static_cast<double>(velocity.axis[axis][index(grid.faces[axis], p)]) - velocity.axis[axis][index(grid.faces[axis], int3{p.x + (axis == 0 ? 1 : 0), p.y + (axis == 1 ? 1 : 0), p.z + (axis == 2 ? 1 : 0)})];
            workspace.residual[i] = static_cast<float>(flux);
            workspace.solution[i] = 0;
        }
    }

    template <bool first>
    __global__ void direction_kernel(PressureLevel fine, float* direction, const IterationState* state) {
        if (state->converged || state->failed) return;
        float beta = first ? 0 : static_cast<float>(state->rz / state->previous_rz);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i        = blockIdx.x * tile_cells + local;
            direction[i] = first ? fine.temporary[i] : fine.temporary[i] + beta * direction[i];
        }
    }

    __global__ void product_kernel(PressureLevel fine, const float* direction, float* result, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        int kind = fine.kind[blockIdx.x];
        if (kind == 0) {
            for (int j = threadIdx.x; j < tile_cells; j += blockDim.x) result[blockIdx.x * tile_cells + j] = 0;
            if (!threadIdx.x) partial[blockIdx.x] = 0;
            return;
        }
        if (kind == 1) product_block<true>(fine, direction, result, partial);
        else product_block<false>(fine, direction, result, partial);
    }

    __global__ void update_kernel(PressureView workspace) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        int root = workspace.block_components[blockIdx.x];
        if (root == -1) return;
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        double alpha = workspace.iteration->rz / workspace.iteration->denominator, sum = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            workspace.solution[i] += alpha * workspace.direction[i];
            float r               = static_cast<float>(workspace.residual[i] - alpha * workspace.product[i]);
            workspace.residual[i] = r;
            if (root >= 0) sum += r;
            else if (workspace.parents[i] >= 0) atomicAdd(workspace.component_sums + workspace.parents[i], static_cast<double>(r));
        }
        if (root >= 0) {
            double total = cub::BlockReduce<double, block_threads>(reduction).Sum(sum);
            if (!threadIdx.x) atomicAdd(workspace.component_sums + root, total);
        }
    }

    __global__ void count_iteration(IterationState* state) {
        if (!state->converged && !state->failed) ++state->iterations;
    }

    __global__ void residual_kernel(Grid grid, Velocity velocity, PressureView workspace) {
        __shared__ typename cub::BlockReduce<double, block_threads>::TempStorage reduction;
        double sum = 0, largest = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i           = blockIdx.x * tile_cells + local;
            int3 p          = coordinate(grid.cells, i);
            double residual = 0;
            if (grid.fluid[i]) {
#pragma unroll
                for (int axis = 0; axis < 3; ++axis)
                    for (int sign = -1; sign <= 1; sign += 2) {
                        int3 face = sign < 0 ? p : int3{p.x + (axis == 0 ? 1 : 0), p.y + (axis == 1 ? 1 : 0), p.z + (axis == 2 ? 1 : 0)};
                        int f     = index(grid.faces[axis], face);
                        residual -= sign * static_cast<double>(velocity.axis[axis][f]);
                        if (!grid.boundary[axis][f]) residual -= workspace.solution[i] - workspace.solution[index(grid.cells, int3{p.x + (axis == 0 ? sign : 0), p.y + (axis == 1 ? sign : 0), p.z + (axis == 2 ? sign : 0)})];
                    }
            }
            largest               = fmax(largest, fabs(residual));
            float r               = static_cast<float>(residual);
            workspace.residual[i] = r;
            sum += static_cast<double>(r) * r;
        }
        double value = cub::BlockReduce<double, block_threads>(reduction).Sum(sum);
        if (!threadIdx.x) workspace.reductions[blockIdx.x] = value;
        __syncthreads();
        value = cub::BlockReduce<double, block_threads>(reduction).Reduce(largest, ::cuda::maximum<>{});
        if (!threadIdx.x) workspace.reductions[grid.cells.blocks + blockIdx.x] = value;
    }

    __global__ void clear_convergence(IterationState* state) {
        state->converged = 0;
    }

    template <int axis>
    __global__ void project_kernel(Grid grid, Velocity velocity, const double* pressure) {
        Layout layout = grid.faces[axis];
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            if (grid.boundary[axis][i]) continue;
            int3 p                 = coordinate(layout, i);
            double gradient        = pressure[index(grid.cells, p)] - pressure[index(grid.cells, int3{p.x + (axis == 0 ? -1 : 0), p.y + (axis == 1 ? -1 : 0), p.z + (axis == 2 ? -1 : 0)})];
            velocity.axis[axis][i] = static_cast<float>(velocity.axis[axis][i] - gradient);
        }
    }

    void precondition(std::span<const PressureLevel> levels, PressureView workspace, ::cuda::stream_ref stream) {
        for (int l = 0; l + 1 < workspace.level_count; ++l) {
            auto fine = levels[l], coarse = levels[l + 1];
            const float* rhs = l == 0 ? workspace.residual : fine.rhs;
            double* partial  = l == 0 ? workspace.reductions : nullptr;
            if (l == 0) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), empty_transfer<true>, fine, coarse, partial, workspace.iteration);
            else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), empty_transfer<false>, fine, coarse, nullptr, workspace.iteration);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), smooth_transfer<false, true>, fine, coarse, rhs, nullptr, workspace.iteration);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), smooth_transfer<false, false>, fine, coarse, rhs, nullptr, workspace.iteration);
        }
        auto bottom = levels[workspace.level_count - 1];
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<block_threads>()), coarsest, bottom, workspace.level_count == 1 ? workspace.residual : bottom.rhs, workspace.iteration);
        for (int l = workspace.level_count - 2; l >= 0; --l) {
            auto fine = levels[l], coarse = levels[l + 1];
            const float* rhs = l == 0 ? workspace.residual : fine.rhs;
            double* partial  = l == 0 ? workspace.reductions : nullptr;
            if (l == 0) {
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), smooth_transfer<true, true, true>, fine, coarse, rhs, partial, workspace.iteration);
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), smooth_transfer<true, false, true>, fine, coarse, rhs, partial, workspace.iteration);
            } else {
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), smooth_transfer<true, true>, fine, coarse, rhs, nullptr, workspace.iteration);
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), smooth_transfer<true, false>, fine, coarse, rhs, nullptr, workspace.iteration);
            }
        }
        if (workspace.level_count == 1) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(bottom.layout.blocks), ::cuda::block_dims<block_threads>()), dot_kernel, bottom, bottom.temporary, workspace.residual, workspace.reductions, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<block_threads>()), finish_dot<DotOperation::update_rz>, workspace.reductions, levels[0].layout.blocks, workspace.iteration, 0, 0);
    }

    void assemble_matrix(std::span<const PressureLevel> levels, Grid grid, PressureView workspace, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), fine_matrix, grid, fine, workspace.parents);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), connect_components, fine, workspace.parents);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), label_components, grid, workspace);
        ::cuda::fill_bytes(stream, ::cuda::std::span(workspace.component_counts, static_cast<std::size_t>(fine.layout.count)), 0);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), component_sums<true>, workspace, nullptr);
        for (int l = 0; l < workspace.level_count; ++l) {
            auto level = levels[l];
            if (l) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<block_threads>()), coarsen_matrix, levels[l - 1], level);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<block_threads>()), classify, level);
        }
    }

    void begin_pressure(std::span<const PressureLevel> levels, Grid grid, Velocity velocity, PressureView workspace, PressureConfig config, float h, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        ::cuda::fill_bytes(stream, ::cuda::std::span(workspace.iteration, 1), 0);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), rhs_kernel, grid, velocity, workspace);
        ::cuda::launch(stream, ::cuda::distribute<block_threads>(fine.layout.blocks), clear_component_sums, workspace, fine.layout.blocks);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), component_sums<false>, workspace, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), center_residual, fine, workspace);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<block_threads>()), finish_dot<DotOperation::initial_norm>, workspace.reductions, fine.layout.blocks, workspace.iteration, config.relative_tolerance, .25f * config.divergence_tolerance * h);
        precondition(levels, workspace, stream);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), direction_kernel<true>, fine, workspace.direction, workspace.iteration);
    }

    void iterate_pressure(std::span<const PressureLevel> levels, PressureView workspace, PressureConfig config, float h, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), product_kernel, fine, workspace.direction, workspace.product, workspace.reductions, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<block_threads>()), finish_dot<DotOperation::denominator>, workspace.reductions, fine.layout.blocks, workspace.iteration, 0, 0);
        ::cuda::launch(stream, ::cuda::distribute<block_threads>(fine.layout.blocks), clear_component_sums, workspace, fine.layout.blocks);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), update_kernel, workspace);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<1>()), count_iteration, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), center_residual, fine, workspace);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<block_threads>()), finish_dot<DotOperation::residual_norm>, workspace.reductions, fine.layout.blocks, workspace.iteration, config.relative_tolerance, .25f * config.divergence_tolerance * h);
        precondition(levels, workspace, stream);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), direction_kernel<false>, fine, workspace.direction, workspace.iteration);
    }

    void check_pressure(std::span<const PressureLevel> levels, Grid grid, Velocity velocity, PressureView workspace, PressureConfig config, float h, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<1>()), clear_convergence, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), residual_kernel, grid, velocity, workspace);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<block_threads>()), finish_dot<DotOperation::residual_norm>, workspace.reductions, fine.layout.blocks, workspace.iteration, config.relative_tolerance, .5f * config.divergence_tolerance * h);
        precondition(levels, workspace, stream);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<block_threads>()), direction_kernel<true>, fine, workspace.direction, workspace.iteration);
    }

    void apply_pressure(Grid grid, Velocity velocity, PressureView workspace, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[0].blocks), ::cuda::block_dims<block_threads>()), project_kernel<0>, grid, velocity, workspace.solution);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[1].blocks), ::cuda::block_dims<block_threads>()), project_kernel<1>, grid, velocity, workspace.solution);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.faces[2].blocks), ::cuda::block_dims<block_threads>()), project_kernel<2>, grid, velocity, workspace.solution);
    }

    Pressure::Pressure(int3 resolution, PressureConfig settings, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool)
        : config(settings),
          stream(stream),
          pool(pool),
          iteration_graph(nullptr, cudaGraphExecDestroy),
          iteration_storage(::cuda::make_buffer<IterationState>(stream, pool, 1, ::cuda::no_init)),
          pressure_storage(stream, pool),
          double_storage(stream, pool),
          index_storage(stream, pool) {
        view.iteration = iteration_storage.data();
        hierarchy.resize(std::bit_width(static_cast<unsigned>(std::max({resolution.x, resolution.y, resolution.z}))));
        int3 size    = resolution;
        float weight = 1;
        for (auto& level : hierarchy) {
            level.layout         = make_layout(size);
            level.regular_weight = weight;
            auto count           = static_cast<std::size_t>(level.layout.count);
            auto& storage        = level_storage.emplace_back(::cuda::make_buffer<float>(stream, pool, count * (view.level_count ? 7 : 6), ::cuda::no_init));
            level.diagonal       = storage.data();
            for (int axis = 0; axis < 3; ++axis) level.weight[axis] = storage.data() + count * (axis + 1);
            level.solution  = storage.data() + count * 4;
            level.temporary = storage.data() + count * 5;
            if (view.level_count) level.rhs = storage.data() + count * 6;
            level.kind = level_kind_storage.emplace_back(::cuda::make_buffer<unsigned char>(stream, pool, level.layout.blocks, ::cuda::no_init)).data();
            ++view.level_count;
            if (std::max({size.x, size.y, size.z}) <= 8) break;
            size = {(size.x + 1) / 2, (size.y + 1) / 2, (size.z + 1) / 2};
            weight *= 4;
        }
        auto count            = static_cast<std::size_t>(make_layout(resolution).count);
        pressure_storage      = ::cuda::make_buffer<float>(stream, pool, count * 3, ::cuda::no_init);
        double_storage        = ::cuda::make_buffer<double>(stream, pool, count * 2 + make_layout(resolution).blocks * 3ull + 3, ::cuda::no_init);
        index_storage         = ::cuda::make_buffer<int>(stream, pool, count * 2 + make_layout(resolution).blocks, ::cuda::no_init);
        view.parents          = index_storage.data();
        view.component_counts = index_storage.data() + count;
        view.block_components = index_storage.data() + count * 2;
        view.component_sums   = double_storage.data();
        view.solution         = double_storage.data() + count;
        view.reductions       = double_storage.data() + count * 2;
        view.residual         = pressure_storage.data();
        view.direction        = pressure_storage.data() + count;
        view.product          = pressure_storage.data() + count * 2;
    }

    void Pressure::build(Grid grid) {
        assemble_matrix(std::span(hierarchy).first(view.level_count), grid, view, stream);
    }

    int Pressure::project(Grid grid, Velocity velocity, float h) {
        auto levels = std::span(hierarchy).first(view.level_count);

        begin_pressure(levels, grid, velocity, view, config, h, stream);
        if (!iteration_graph) {
            cudaGraph_t captured{};
            if (auto cuda_result = cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            for (int i = 0; i < 4; ++i) iterate_pressure(levels, view, config, h, stream);
            if (auto cuda_result = cudaStreamEndCapture(stream.get(), &captured); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph{captured, cudaGraphDestroy};
            cudaGraphExec_t executable{};
            if (auto cuda_result = cudaGraphInstantiate(&executable, graph.get(), 0); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            iteration_graph.reset(executable);
        }
        IterationState result{};
        for (;;) {
            int remaining = config.max_iterations - result.iterations;
            if (remaining < 4)
                for (int i = 0; i < remaining; ++i) iterate_pressure(levels, view, config, h, stream);
            else if (auto cuda_result = cudaGraphLaunch(iteration_graph.get(), stream.get()); cuda_result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(cuda_result));
            ::cuda::copy_bytes(stream, ::cuda::std::span(view.iteration, 1), ::cuda::std::span(&result, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            if (result.converged) {
                check_pressure(levels, grid, velocity, view, config, h, stream);
                ::cuda::copy_bytes(stream, ::cuda::std::span(view.iteration, 1), ::cuda::std::span(&result, 1), {.src_access_order = ::cuda::source_access_order::stream});
                stream.sync();
                if (result.converged) break;
            }
            if (result.failed || result.iterations >= config.max_iterations) throw std::runtime_error("LFM pressure did not converge: iterations=" + std::to_string(result.iterations) + ", residual=" + std::to_string(result.max_residual / h));
        }
        apply_pressure(grid, velocity, view, stream);
        return result.iterations;
    }

}
