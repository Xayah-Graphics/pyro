#include "cuda.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>

namespace pyro::cuda::lfm {
    __global__ void fine_matrix(Grid grid, PressureLevel level, int* parents) {
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i          = blockIdx.x * tileCells + local;
            int3 p         = coordinate(level.layout, i);
            parents[i]     = i;
            float diagonal = 0;
            for (int axis = 0; axis < 3; ++axis) {
                bool negative         = grid.fluid[i] && !grid.boundary[axis][index(grid.faces[axis], p)];
                level.weight[axis][i] = negative ? 1 : 0;
                if (grid.fluid[i]) diagonal += negative + !grid.boundary[axis][index(grid.faces[axis], offset(p, axis, 1))];
            }
            level.diagonal[i] = diagonal;
        }
    }
    __global__ void connect_components(PressureLevel level, int* parents) {
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i  = blockIdx.x * tileCells + local;
            int3 p = coordinate(level.layout, i);
            for (int axis = 0; axis < 3; ++axis) {
                if (!level.weight[axis][i]) continue;
                int j = index(level.layout, offset(p, axis, -1));
                int a = component_root(parents, i), b = component_root(parents, j);
                while (a != b) {
                    int high = max(a, b), low = min(a, b);
                    int old = atomicCAS(parents + high, high, low);
                    if (old == high) break;
                    a = component_root(parents, old);
                    b = component_root(parents, low);
                }
            }
        }
    }
    __global__ void label_components(Grid grid, PressureWorkspace workspace) {
        __shared__ typename cub::BlockReduce<int, blockThreads>::TempStorage reduction;
        int lower = 0x7fffffff, upper = -1;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i    = blockIdx.x * tileCells + local;
            int root = grid.fluid[i] ? component_root(workspace.parents, i) : -1;
            atomicExch(workspace.parents + i, root);
            if (root >= 0) lower = min(lower, root);
            upper = max(upper, root);
        }
        int lo = cub::BlockReduce<int, blockThreads>(reduction).Reduce(lower, ::cuda::minimum<>{});
        if (!threadIdx.x) workspace.blockComponents[blockIdx.x] = lo;
        __syncthreads();
        int hi = cub::BlockReduce<int, blockThreads>(reduction).Reduce(upper, ::cuda::maximum<>{});
        if (!threadIdx.x) workspace.blockComponents[blockIdx.x] = hi < 0 ? -1 : hi == workspace.blockComponents[blockIdx.x] ? hi : -2;
    }
    template <bool count>
    __global__ void component_sums(PressureWorkspace workspace, const IterationState* state) {
        if constexpr (!count)
            if (state->converged || state->failed) return;
        __shared__ typename cub::BlockReduce<double, blockThreads>::TempStorage reduction;
        int root = workspace.blockComponents[blockIdx.x];
        if (root == -1) return;
        double sum = 0;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i = blockIdx.x * tileCells + local, component = workspace.parents[i];
            if (component < 0) continue;
            double value = count ? 1 : workspace.residual[i];
            if (root >= 0) sum += value;
            else {
                if constexpr (count) atomicAdd(workspace.componentCounts + component, 1);
                else atomicAdd(workspace.componentSums + component, value);
            }
        }
        if (root >= 0) {
            double total = cub::BlockReduce<double, blockThreads>(reduction).Sum(sum);
            if (!threadIdx.x) {
                if constexpr (count) atomicAdd(workspace.componentCounts + root, static_cast<int>(total));
                else atomicAdd(workspace.componentSums + root, total);
            }
        }
    }
    __global__ void center_residual(PressureLevel level, PressureWorkspace workspace) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        __shared__ typename cub::BlockReduce<double, blockThreads>::TempStorage reduction;
        double sum = 0, largest = 0;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i = blockIdx.x * tileCells + local, root = workspace.parents[i];
            float r               = root >= 0 ? static_cast<float>(workspace.residual[i] - workspace.componentSums[root] / workspace.componentCounts[root]) : 0;
            workspace.residual[i] = r;
            sum += static_cast<double>(r) * r;
            largest = fmax(largest, fabs(static_cast<double>(r)));
        }
        double value = cub::BlockReduce<double, blockThreads>(reduction).Sum(sum);
        if (!threadIdx.x) workspace.reductions[blockIdx.x] = value;
        __syncthreads();
        value = cub::BlockReduce<double, blockThreads>(reduction).Reduce(largest, ::cuda::maximum<>{});
        if (!threadIdx.x) workspace.reductions[level.layout.blocks + blockIdx.x] = value;
    }
    __global__ void clear_component_sums(PressureWorkspace workspace, int blocks) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        int block = blockIdx.x * blockDim.x + threadIdx.x;
        if (block >= blocks) return;
        int root = workspace.blockComponents[block];
        if (root >= 0) atomicExch(reinterpret_cast<unsigned long long*>(workspace.componentSums + root), 0ull);
        else if (root == -2)
            for (int j = 0; j < tileCells; ++j) {
                int component = workspace.parents[block * tileCells + j];
                if (component >= 0) atomicExch(reinterpret_cast<unsigned long long*>(workspace.componentSums + component), 0ull);
            }
    }
    __global__ void coarsen_matrix(PressureLevel fine, PressureLevel coarse) {
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i  = blockIdx.x * tileCells + local;
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
                            for (int axis = 0; axis < 3; ++axis) {
                                if (!parity[axis]) {
                                    weights[axis] += fine.weight[axis][j];
                                    diagonal += fine.weight[axis][j];
                                } else diagonal += read(fine.layout, fine.weight[axis], offset(q, axis, 1));
                            }
                        }
            }
            coarse.diagonal[i] = diagonal;
            for (int axis = 0; axis < 3; ++axis) coarse.weight[axis][i] = weights[axis];
        }
    }
    __global__ void classify(PressureLevel level) {
        __shared__ typename cub::BlockReduce<int, blockThreads>::TempStorage reduction;
        int3 origin = coordinate(level.layout, blockIdx.x * tileCells);
        int active = 0, irregular = 0;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) active |= level.diagonal[blockIdx.x * tileCells + local] > 0;
        // The fused smoother reads a two-cell halo. Classify that full footprint.
        for (int j = threadIdx.x; j < 1728; j += blockDim.x) {
            int3 p{origin.x + j % 12 - 2, origin.y + j / 12 % 12 - 2, origin.z + j / 144 - 2};
            if (!contains(level.layout, p)) irregular = 1;
            else {
                int i = index(level.layout, p);
                irregular |= level.diagonal[i] != 6 * level.regularWeight;
                for (int axis = 0; axis < 3; ++axis) irregular |= level.weight[axis][i] != level.regularWeight;
            }
        }
        int any = cub::BlockReduce<int, blockThreads>(reduction).Reduce(active, ::cuda::maximum<>{});
        if (!threadIdx.x) level.kind[blockIdx.x] = any ? 1 : 0;
        __syncthreads();
        int general = cub::BlockReduce<int, blockThreads>(reduction).Reduce(irregular, ::cuda::maximum<>{});
        if (!threadIdx.x && level.kind[blockIdx.x] && general) level.kind[blockIdx.x] = 2;
    }

    // Pre: zero guess -> red/black GS -> residual/restriction.
    // Post: prolong -> black/red GS -> optional r.z reduction.
    // Halo colors are recomputed locally. Post writes a distinct buffer so a CTA
    // never observes another CTA's partially updated iterate.
    template <bool post, bool regular>
    __global__ void smooth_transfer(PressureLevel fine, PressureLevel coarse, const float* rhs, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        if (fine.kind[blockIdx.x] != (regular ? 1 : 2)) return;
        __shared__ float x[1728], b[1728];
        __shared__ float diagonal[regular ? 1 : 1728], weight[regular ? 1 : 3][regular ? 1 : 1728];
        __shared__ typename cub::BlockReduce<double, blockThreads>::TempStorage reduction;
        int3 origin = coordinate(fine.layout, blockIdx.x * tileCells);
        for (int j = threadIdx.x; j < 1728; j += blockDim.x) {
            int3 p{origin.x + j % 12 - 2, origin.y + j / 12 % 12 - 2, origin.z + j / 144 - 2};
            bool valid = contains(fine.layout, p);
            int i      = valid ? index(fine.layout, p) : 0;
            float d    = regular ? 6 * fine.regularWeight : valid ? fine.diagonal[i] : 0;
            b[j]       = valid ? rhs[i] : 0;
            if constexpr (!regular) {
                diagonal[j] = d;
                for (int axis = 0; axis < 3; ++axis) weight[axis][j] = valid ? fine.weight[axis][i] : 0;
            }
            if constexpr (post) x[j] = d > 0 ? fine.solution[i] + 2 * coarse.temporary[index(coarse.layout, {p.x / 2, p.y / 2, p.z / 2})] : 0;
            else x[j] = d > 0 && !((p.x + p.y + p.z) & 1) ? b[j] / d : 0;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < 1000; j += blockDim.x) {
            int px = j % 10 + 1, py = j / 10 % 10 + 1, pz = j / 100 + 1;
            int s = px + 12 * py + 144 * pz;
            if (!((px + py + pz) & 1)) continue;
            float d     = regular ? 6 * fine.regularWeight : diagonal[s];
            float value = b[s];
            for (int axis = 0; axis < 3; ++axis) {
                int stride = axis == 0 ? 1 : axis == 1 ? 12 : 144;
                float a    = regular ? fine.regularWeight : weight[axis][s];
                float c    = regular ? fine.regularWeight : weight[axis][s + stride];
                value += a * x[s - stride] + c * x[s + stride];
            }
            x[s] = d > 0 ? value / d : 0;
        }
        __syncthreads();
        double sum = 0;
        for (int j = threadIdx.x; j < tileCells; j += blockDim.x) {
            int px = (j & 7) + 2, py = ((j >> 3) & 7) + 2, pz = (j >> 6) + 2;
            int s = px + 12 * py + 144 * pz, i = blockIdx.x * tileCells + j;
            float d         = regular ? 6 * fine.regularWeight : diagonal[s];
            float neighbors = 0;
            for (int axis = 0; axis < 3; ++axis) {
                int stride = axis == 0 ? 1 : axis == 1 ? 12 : 144;
                float a    = regular ? fine.regularWeight : weight[axis][s];
                float c    = regular ? fine.regularWeight : weight[axis][s + stride];
                neighbors += a * x[s - stride] + c * x[s + stride];
            }
            if constexpr (post) {
                float result      = !((px + py + pz) & 1) && d > 0 ? (b[s] + neighbors) / d : x[s];
                fine.temporary[i] = result;
                sum += static_cast<double>(result) * b[s];
            } else {
                fine.solution[i] = x[s];
                b[s]             = d > 0 ? b[s] - (d * x[s] - neighbors) : 0;
            }
        }
        __syncthreads();
        if constexpr (post) {
            if (partial) {
                double dot = cub::BlockReduce<double, blockThreads>(reduction).Sum(sum);
                if (!threadIdx.x) {
                    partial[blockIdx.x]                      = dot;
                    partial[fine.layout.blocks + blockIdx.x] = 0;
                }
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
    __global__ void empty_transfer(PressureLevel fine, PressureLevel coarse, double* partial, const IterationState* state) {
        if (state->converged || state->failed || fine.kind[blockIdx.x]) return;
        int3 origin = coordinate(fine.layout, blockIdx.x * tileCells);
        for (int j = threadIdx.x; j < tileCells; j += blockDim.x) {
            int i            = blockIdx.x * tileCells + j;
            fine.solution[i] = fine.temporary[i] = 0;
        }
        for (int j = threadIdx.x; j < 64; j += blockDim.x) {
            int3 p{origin.x / 2 + (j & 3), origin.y / 2 + ((j >> 2) & 3), origin.z / 2 + (j >> 4)};
            if (contains(coarse.layout, p)) coarse.rhs[index(coarse.layout, p)] = 0;
        }
        if (!threadIdx.x && partial) partial[blockIdx.x] = partial[fine.layout.blocks + blockIdx.x] = 0;
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
                    for (int axis = 0; axis < 3; ++axis) {
                        int stride = axis == 0 ? 1 : axis == 1 ? nx : nx * ny;
                        if (p[axis] > 0) value += w[axis][j] * x[j - stride];
                        if (p[axis] + 1 < size[axis]) value += w[axis][j + stride] * x[j + stride];
                    }
                    x[j] = value / d[j];
                }
                __syncthreads();
            }
        for (int j = threadIdx.x; j < tileCells; j += blockDim.x) level.temporary[j] = 0;
        __syncthreads();
        for (int j = threadIdx.x; j < count; j += blockDim.x) level.temporary[index(level.layout, {j % nx, j / nx % ny, j / (nx * ny)})] = x[j];
    }
    __global__ void dot_kernel(PressureLevel level, const float* a, const float* b, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ typename cub::BlockReduce<double, blockThreads>::TempStorage reduction;
        double sum = 0;
        for (int j = threadIdx.x; j < tileCells; j += blockDim.x) {
            int i = blockIdx.x * tileCells + j;
            sum += static_cast<double>(a[i]) * b[i];
        }
        double value = cub::BlockReduce<double, blockThreads>(reduction).Sum(sum);
        if (!threadIdx.x) partial[blockIdx.x] = value;
        if (!threadIdx.x) partial[level.layout.blocks + blockIdx.x] = 0;
    }
    void precondition(PressureWorkspace workspace, cudaStream_t stream) {
        for (int l = 0; l + 1 < workspace.levelCount; ++l) {
            auto fine = workspace.levels[l], coarse = workspace.levels[l + 1];
            const float* rhs = l == 0 ? workspace.residual : fine.rhs;
            double* partial  = l == 0 ? workspace.reductions : nullptr;
            empty_transfer<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, coarse, partial, workspace.iteration);
            smooth_transfer<false, true><<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, coarse, rhs, nullptr, workspace.iteration);
            smooth_transfer<false, false><<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, coarse, rhs, nullptr, workspace.iteration);
        }
        auto bottom = workspace.levels[workspace.levelCount - 1];
        coarsest<<<1, blockThreads, 0, stream>>>(bottom, workspace.levelCount == 1 ? workspace.residual : bottom.rhs, workspace.iteration);
        for (int l = workspace.levelCount - 2; l >= 0; --l) {
            auto fine = workspace.levels[l], coarse = workspace.levels[l + 1];
            const float* rhs = l == 0 ? workspace.residual : fine.rhs;
            double* partial  = l == 0 ? workspace.reductions : nullptr;
            smooth_transfer<true, true><<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, coarse, rhs, partial, workspace.iteration);
            smooth_transfer<true, false><<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, coarse, rhs, partial, workspace.iteration);
        }
        if (workspace.levelCount == 1) dot_kernel<<<bottom.layout.blocks, blockThreads, 0, stream>>>(bottom, bottom.temporary, workspace.residual, workspace.reductions, workspace.iteration);
        finish_dot<blockThreads><<<1, blockThreads, 0, stream>>>(workspace.reductions, workspace.levels[0].layout.blocks, workspace.iteration, 0, 0, 0);
    }
    __global__ void rhs_kernel(Grid grid, Velocity velocity, PressureWorkspace workspace) {
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i       = blockIdx.x * tileCells + local;
            int3 p      = coordinate(grid.cells, i);
            double flux = 0;
            if (grid.fluid[i])
                for (int axis = 0; axis < 3; ++axis) flux += static_cast<double>(velocity.axis[axis][index(grid.faces[axis], p)]) - velocity.axis[axis][index(grid.faces[axis], offset(p, axis, 1))];
            workspace.residual[i] = static_cast<float>(flux);
            workspace.solution[i] = 0;
        }
    }
    __global__ void direction_kernel(PressureLevel fine, float* direction, const IterationState* state, bool first) {
        if (state->converged || state->failed) return;
        float beta = first ? 0 : static_cast<float>(state->rz / state->previousRz);
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i        = blockIdx.x * tileCells + local;
            direction[i] = first ? fine.temporary[i] : fine.temporary[i] + beta * direction[i];
        }
    }
    __global__ void product_kernel(PressureLevel fine, const float* direction, float* result, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ float x[1000];
        __shared__ typename cub::BlockReduce<double, blockThreads>::TempStorage reduction;
        int3 origin  = coordinate(fine.layout, blockIdx.x * tileCells);
        bool regular = fine.kind[blockIdx.x] == 1;
        for (int j = threadIdx.x; j < 1000; j += blockDim.x) x[j] = read(fine.layout, direction, {origin.x + j % 10 - 1, origin.y + j / 10 % 10 - 1, origin.z + j / 100 - 1});
        __syncthreads();
        double sum = 0;
        for (int j = threadIdx.x; j < tileCells; j += blockDim.x) {
            int i = blockIdx.x * tileCells + j;
            int s = (j & 7) + 1 + 10 * (((j >> 3) & 7) + 1) + 100 * ((j >> 6) + 1);
            int3 p{origin.x + (j & 7), origin.y + ((j >> 3) & 7), origin.z + (j >> 6)};
            float value = 0;
            if (fine.diagonal[i] > 0) {
                for (int axis = 0; axis < 3; ++axis) {
                    int stride = axis == 0 ? 1 : axis == 1 ? 10 : 100;
                    float a    = regular ? fine.regularWeight : fine.weight[axis][i];
                    float b    = regular ? fine.regularWeight : read(fine.layout, fine.weight[axis], offset(p, axis, 1));
                    value += a * (x[s] - x[s - stride]) + b * (x[s] - x[s + stride]);
                }
            }
            result[i] = value;
            sum += static_cast<double>(direction[i]) * value;
        }
        double value = cub::BlockReduce<double, blockThreads>(reduction).Sum(sum);
        if (!threadIdx.x) {
            partial[blockIdx.x]                      = value;
            partial[fine.layout.blocks + blockIdx.x] = 0;
        }
    }
    __global__ void update_kernel(PressureWorkspace workspace) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        int root = workspace.blockComponents[blockIdx.x];
        if (root == -1) return;
        __shared__ typename cub::BlockReduce<double, blockThreads>::TempStorage reduction;
        double alpha = workspace.iteration->rz / workspace.iteration->denominator, sum = 0;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i = blockIdx.x * tileCells + local;
            workspace.solution[i] += alpha * workspace.direction[i];
            float r               = static_cast<float>(workspace.residual[i] - alpha * workspace.product[i]);
            workspace.residual[i] = r;
            if (root >= 0) sum += r;
            else if (workspace.parents[i] >= 0) atomicAdd(workspace.componentSums + workspace.parents[i], static_cast<double>(r));
        }
        if (root >= 0) {
            double total = cub::BlockReduce<double, blockThreads>(reduction).Sum(sum);
            if (!threadIdx.x) atomicAdd(workspace.componentSums + root, total);
        }
    }
    __global__ void count_iteration(IterationState* state) {
        if (!state->converged && !state->failed) ++state->iterations;
    }
    __global__ void residual_kernel(Grid grid, Velocity velocity, PressureWorkspace workspace) {
        __shared__ typename cub::BlockReduce<double, blockThreads>::TempStorage reduction;
        double sum = 0, largest = 0;
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i           = blockIdx.x * tileCells + local;
            int3 p          = coordinate(grid.cells, i);
            double residual = 0;
            if (grid.fluid[i]) {
                for (int axis = 0; axis < 3; ++axis)
                    for (int sign = -1; sign <= 1; sign += 2) {
                        int3 face = sign < 0 ? p : offset(p, axis, 1);
                        int f     = index(grid.faces[axis], face);
                        residual -= sign * static_cast<double>(velocity.axis[axis][f]);
                        if (!grid.boundary[axis][f]) residual -= workspace.solution[i] - workspace.solution[index(grid.cells, offset(p, axis, sign))];
                    }
            }
            // Rebuild the physical divergence from face fluxes in FP64, including
            // every cell of each Neumann component; no pinned equation is omitted.
            largest               = fmax(largest, fabs(residual));
            float r               = static_cast<float>(residual);
            workspace.residual[i] = r;
            sum += static_cast<double>(r) * r;
        }
        double value = cub::BlockReduce<double, blockThreads>(reduction).Sum(sum);
        if (!threadIdx.x) workspace.reductions[blockIdx.x] = value;
        __syncthreads();
        value = cub::BlockReduce<double, blockThreads>(reduction).Reduce(largest, ::cuda::maximum<>{});
        if (!threadIdx.x) workspace.reductions[grid.cells.blocks + blockIdx.x] = value;
    }
    __global__ void clear_convergence(IterationState* state) {
        state->converged = 0;
    }
    __global__ void project_kernel(Grid grid, Velocity velocity, const double* pressure, int axis) {
        Layout layout = grid.faces[axis];
        for (int local = threadIdx.x; local < tileCells; local += blockDim.x) {
            int i = blockIdx.x * tileCells + local;
            if (grid.boundary[axis][i]) continue;
            int3 p                 = coordinate(layout, i);
            double gradient        = pressure[index(grid.cells, p)] - pressure[index(grid.cells, offset(p, axis, -1))];
            velocity.axis[axis][i] = static_cast<float>(velocity.axis[axis][i] - gradient);
        }
    }
    void build_pressure(Grid grid, PressureWorkspace workspace, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        fine_matrix<<<fine.layout.blocks, blockThreads, 0, stream>>>(grid, fine, workspace.parents);
        connect_components<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, workspace.parents);
        label_components<<<fine.layout.blocks, blockThreads, 0, stream>>>(grid, workspace);
        cuda_check(cudaMemsetAsync(workspace.componentCounts, 0, fine.layout.count * sizeof(int), stream));
        component_sums<true><<<fine.layout.blocks, blockThreads, 0, stream>>>(workspace, nullptr);
        for (int l = 0; l < workspace.levelCount; ++l) {
            auto level = workspace.levels[l];
            if (l) coarsen_matrix<<<level.layout.blocks, blockThreads, 0, stream>>>(workspace.levels[l - 1], level);
            classify<<<level.layout.blocks, blockThreads, 0, stream>>>(level);
        }
    }
    void begin_pressure(Grid grid, Velocity velocity, PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        cuda_check(cudaMemsetAsync(workspace.iteration, 0, sizeof(IterationState), stream));
        rhs_kernel<<<fine.layout.blocks, blockThreads, 0, stream>>>(grid, velocity, workspace);
        clear_component_sums<<<(fine.layout.blocks + blockThreads - 1) / blockThreads, blockThreads, 0, stream>>>(workspace, fine.layout.blocks);
        component_sums<false><<<fine.layout.blocks, blockThreads, 0, stream>>>(workspace, workspace.iteration);
        center_residual<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, workspace);
        finish_dot<blockThreads><<<1, blockThreads, 0, stream>>>(workspace.reductions, fine.layout.blocks, workspace.iteration, 2, config.pressureRelativeTolerance, .25f * config.divergenceTolerance * config.cellSize);
        precondition(workspace, stream);
        direction_kernel<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, workspace.direction, workspace.iteration, true);
    }
    void iterate_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        product_kernel<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, workspace.direction, workspace.product, workspace.reductions, workspace.iteration);
        finish_dot<blockThreads><<<1, blockThreads, 0, stream>>>(workspace.reductions, fine.layout.blocks, workspace.iteration, 1, 0, 0);
        clear_component_sums<<<(fine.layout.blocks + blockThreads - 1) / blockThreads, blockThreads, 0, stream>>>(workspace, fine.layout.blocks);
        update_kernel<<<fine.layout.blocks, blockThreads, 0, stream>>>(workspace);
        count_iteration<<<1, 1, 0, stream>>>(workspace.iteration);
        center_residual<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, workspace);
        finish_dot<blockThreads><<<1, blockThreads, 0, stream>>>(workspace.reductions, fine.layout.blocks, workspace.iteration, 3, config.pressureRelativeTolerance, .25f * config.divergenceTolerance * config.cellSize);
        precondition(workspace, stream);
        direction_kernel<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, workspace.direction, workspace.iteration, false);
    }
    void check_pressure(Grid grid, Velocity velocity, PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        clear_convergence<<<1, 1, 0, stream>>>(workspace.iteration);
        residual_kernel<<<fine.layout.blocks, blockThreads, 0, stream>>>(grid, velocity, workspace);
        finish_dot<blockThreads><<<1, blockThreads, 0, stream>>>(workspace.reductions, fine.layout.blocks, workspace.iteration, 3, config.pressureRelativeTolerance, .5f * config.divergenceTolerance * config.cellSize);
        precondition(workspace, stream);
        direction_kernel<<<fine.layout.blocks, blockThreads, 0, stream>>>(fine, workspace.direction, workspace.iteration, true);
    }
    void apply_pressure(Grid grid, Velocity velocity, PressureWorkspace workspace, cudaStream_t stream) {
        for (int axis = 0; axis < 3; ++axis) project_kernel<<<grid.faces[axis].blocks, blockThreads, 0, stream>>>(grid, velocity, workspace.solution, axis);
    }
} // namespace pyro::cuda::lfm
