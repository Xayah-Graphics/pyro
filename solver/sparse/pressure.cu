#include "cuda.cuh"

namespace pyro::cuda {
    __device__ float matrix_product(PressureLevel level, const float* x, int i) {
        // Flux differences preserve the constant nullspace in FP32. Expanding
        // diag*x - sum(face*x_neighbor) loses accuracy for smooth pressures.
        float result = level.boundary[i] * x[i];
        for (int axis = 0; axis < 3; ++axis) {
            int a = neighbor(level.layout, i, axis, -1), b = neighbor(level.layout, i, axis, 1);
            result += level.face[axis][i] * (x[i] - value(x, a));
            result += value(level.face[axis], b) * (x[i] - value(x, b));
        }
        return result;
    }

    template <bool rebuild>
    __global__ void assemble_fine(Grid grid, Fields fields, PressureLevel level, float h, int* parents, int* open, unsigned char* anchors, float* solution, float* residual) {
        int id      = grid.layout.active[blockIdx.x];
        float scale = 1.0f / (h * h);
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i          = id * blockCells + local;
            float diagonal = 0.0f, boundary = 0.0f, divergence = -grid.solidDivergence[i];
            bool isOpen = false;
            if constexpr (rebuild) {
                parents[i] = i;
                open[i]    = 0;
                anchors[i] = 0;
            }
            solution[i] = 0.0f;
            for (int axis = 0; axis < 3; ++axis) {
                int previous = neighbor(grid.layout, i, axis, -1);
                if constexpr (rebuild) level.face[axis][i] = grid.fluid[i] && previous >= 0 && grid.fluid[previous] ? grid.aperture[axis][i] * scale : 0.0f;
                for (int sign = -1; sign <= 1; sign += 2) {
                    int adjacent        = neighbor(grid.layout, i, axis, sign);
                    int face            = sign < 0 ? i : adjacent;
                    float aperture      = value(grid.aperture[axis], face, 1.0f);
                    float fluidVelocity = value(fields.velocity[axis], face);
                    float solidVelocity = value(grid.solidVelocity[axis], face);
                    divergence += sign * (aperture * fluidVelocity + (1.0f - aperture) * solidVelocity) / h;
                    if constexpr (rebuild) {
                        if (adjacent >= 0 && grid.fluid[adjacent]) diagonal += aperture * scale;
                        else if (adjacent < 0 || !(grid.roles[adjacent >> 9] & 2u)) {
                            diagonal += 2.0f * aperture * scale;
                            boundary += 2.0f * aperture * scale;
                            isOpen = isOpen || aperture > 0.0f;
                        }
                    }
                }
            }
            if constexpr (rebuild) {
                level.diagonal[i]  = grid.fluid[i] ? diagonal : 0.0f;
                level.boundary[i]  = grid.fluid[i] ? boundary : 0.0f;
                level.temporary[i] = isOpen && grid.fluid[i] ? 1.0f : 0.0f;
            }
            level.rhs[i] = grid.fluid[i] && !anchors[i] ? -divergence : 0.0f;
            residual[i]  = level.rhs[i];
        }
    }

    __device__ int root(int* parents, int i) {
        int p = atomicAdd(parents + i, 0);
        while (p != i) {
            // Parent labels only decrease during concurrent union and path splitting.
            int next = atomicAdd(parents + p, 0);
            atomicMin(parents + i, next);
            i = p;
            p = next;
        }
        return i;
    }

    __global__ void connect_components(PressureLevel level, const unsigned char* fluid, int* parents) {
        int id = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            if (!fluid[i]) continue;
            for (int axis = 0; axis < 3; ++axis) {
                if (level.face[axis][i] <= 0.0f) continue;
                int j = neighbor(level.layout, i, axis, -1);
                int a = root(parents, i), b = root(parents, j);
                while (a != b) {
                    int high = max(a, b), low = min(a, b);
                    int old = atomicCAS(parents + high, high, low);
                    if (old == high) break;
                    a = root(parents, old);
                    b = root(parents, low);
                }
            }
        }
    }

    __global__ void mark_components(PressureLevel level, int* parents, int* open) {
        int id = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            int r = root(parents, i);
            atomicExch(parents + i, r);
            if (level.temporary[i] > 0.0f) atomicExch(open + r, 1);
        }
    }

    __global__ void anchor_components(PressureLevel level, const unsigned char* fluid, int* parents, const int* open, unsigned char* anchors, float* residual) {
        int id = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i       = id * blockCells + local;
            int r       = parents[i];
            bool anchor = fluid[i] && !open[r] && r == i;
            anchors[i]  = anchor;
            if (anchor) {
                level.diagonal[i] = 1.0f;
                level.boundary[i] = 1.0f;
                level.rhs[i]      = 0.0f;
                residual[i]       = 0.0f;
            }
        }
    }

    __global__ void anchor_faces(PressureLevel level, const unsigned char* anchors) {
        int id = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            for (int axis = 0; axis < 3; ++axis) {
                int j = neighbor(level.layout, i, axis, -1);
                if (anchors[i] || (j >= 0 && anchors[j])) {
                    float weight = level.face[axis][i];
                    if (!anchors[i]) atomicAdd(level.boundary + i, weight);
                    if (j >= 0 && !anchors[j]) atomicAdd(level.boundary + j, weight);
                    level.face[axis][i] = 0.0f;
                }
            }
        }
    }

    __global__ void coarse_keys(Layout fine, Layout coarse) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= fine.blocks) return;
        int3 p      = fine.coordinates[fine.active[index]];
        auto wanted = key({p.x >> 1, p.y >> 1, p.z >> 1});
        int slot    = hash_slot(wanted, coarse.tableCapacity);
        for (;;) {
            auto old = atomicCAS(coarse.keys + slot, emptyKey, wanted);
            if (old == emptyKey || old == wanted) return;
            slot = (slot + 1) & (coarse.tableCapacity - 1);
        }
    }

    __global__ void coarse_indices(Layout coarse) {
        int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (slot >= coarse.tableCapacity || coarse.keys[slot] == emptyKey) return;
        int id                 = atomicAdd(coarse.count, 1);
        coarse.indices[slot]   = id;
        coarse.active[id]      = id;
        coarse.coordinates[id] = coordinate(coarse.keys[slot]);
    }

    __global__ void coarsen_matrix(PressureLevel fine, PressureLevel coarse) {
        int id = coarse.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i          = id * blockCells + local;
            int3 c         = cell_coordinate(coarse.layout, i);
            float diagonal = 0.0f, boundary = 0.0f;
            float faces[3]{};
            // P is piecewise constant, R = transpose(P). Sum only external
            // faces; do not subtract large internal diagonal contributions.
            for (int z = 0; z < 2; ++z)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 2; ++x) {
                        int j = cell_index(fine.layout, {2 * c.x + x, 2 * c.y + y, 2 * c.z + z});
                        if (j < 0) continue;
                        boundary += fine.boundary[j];
                        int parity[]{x, y, z};
                        for (int axis = 0; axis < 3; ++axis) {
                            if (parity[axis]) diagonal += value(fine.face[axis], neighbor(fine.layout, j, axis, 1));
                            else {
                                faces[axis] += fine.face[axis][j];
                                diagonal += fine.face[axis][j];
                            }
                        }
                    }
            coarse.diagonal[i] = diagonal + boundary;
            coarse.boundary[i] = boundary;
            for (int axis = 0; axis < 3; ++axis) coarse.face[axis][i] = faces[axis];
        }
    }

    __global__ void zero_solution(PressureLevel level, const IterationState* state) {
        if (state->converged || state->failed) return;
        int id = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) level.solution[id * blockCells + local] = 0.0f;
    }

    __global__ void smooth(PressureLevel level, const float* rhs, int color, const IterationState* state) {
        if (state->converged || state->failed) return;
        int id = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            if (((local + (local >> 3) + (local >> 6)) & 1) != color) continue;
            int i          = id * blockCells + local;
            float diagonal = level.diagonal[i];
            if (diagonal > 0.0f) level.solution[i] += (rhs[i] - matrix_product(level, level.solution, i)) / diagonal;
        }
    }

    __global__ void restrict_residual(PressureLevel fine, PressureLevel coarse, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        int id = coarse.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i     = id * blockCells + local;
            int3 c    = cell_coordinate(coarse.layout, i);
            float sum = 0.0f;
            for (int z = 0; z < 2; ++z)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 2; ++x) {
                        int j = cell_index(fine.layout, {2 * c.x + x, 2 * c.y + y, 2 * c.z + z});
                        if (j >= 0 && fine.diagonal[j] > 0.0f) sum += rhs[j] - matrix_product(fine, fine.solution, j);
                    }
            coarse.rhs[i] = sum;
        }
    }

    __global__ void prolongate(PressureLevel fine, PressureLevel coarse, const IterationState* state) {
        if (state->converged || state->failed) return;
        int id = fine.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i  = id * blockCells + local;
            int3 c = cell_coordinate(fine.layout, i);
            if (fine.diagonal[i] > 0.0f) fine.solution[i] += value(coarse.solution, cell_index(coarse.layout, {c.x >> 1, c.y >> 1, c.z >> 1}));
        }
    }

    __global__ void solve_coarsest(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        // A single CTA owns the entire coarse solve, including its global barriers.
        // Symmetric relaxation remains a preconditioner; outer PCG checks the true residual.
        for (int sweep = 0; sweep < 8; ++sweep)
            for (int phase = 0; phase < 4; ++phase) {
                int color = phase == 0 || phase == 3 ? 0 : 1;
                for (int j = threadIdx.x; j < level.layout.blocks * blockCells; j += blockDim.x) {
                    int local = j & 511;
                    if (((local + (local >> 3) + (local >> 6)) & 1) != color) continue;
                    int i          = level.layout.active[j >> 9] * blockCells + local;
                    float diagonal = level.diagonal[i];
                    if (diagonal > 0.0f) level.solution[i] += (rhs[i] - matrix_product(level, level.solution, i)) / diagonal;
                }
                __syncthreads();
            }
    }

    __global__ void dot_blocks(PressureLevel level, const float* a, const float* b, double* partial, const IterationState* state, bool maximum) {
        if (state->converged || state->failed) return;
        __shared__ double sums[threads];
        __shared__ float maxima[threads];
        int id        = level.layout.active[blockIdx.x];
        double sum    = 0.0;
        float largest = 0.0f;
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            if (maximum) largest = fmaxf(largest, fabsf(a[i]));
        }
        sums[threadIdx.x]   = sum;
        maxima[threadIdx.x] = largest;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
                maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partial[blockIdx.x]                       = sums[0];
            partial[level.layout.blocks + blockIdx.x] = maxima[0];
        }
    }

    __global__ void finish_dot(double* partial, int count, IterationState* state, int operation, float relativeTolerance, float absoluteTolerance) {
        if (state->converged || state->failed) return;
        __shared__ double sums[threads], maxima[threads];
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
            state->previousRz = state->rz;
            state->rz         = sums[0];
        } else if (operation == 1) {
            state->denominator = sums[0];
            if (sums[0] <= 0.0) state->failed = 1;
        } else {
            if (operation == 2) state->initialNorm = sums[0];
            state->residualNorm = sums[0];
            state->maxResidual  = static_cast<float>(maxima[0]);
            bool relative       = sums[0] <= state->initialNorm * relativeTolerance * relativeTolerance;
            state->converged    = maxima[0] <= absoluteTolerance && (relative || maxima[0] <= absoluteTolerance * 0.1f);
        }
    }

    __global__ void search_direction(PressureLevel level, float* direction, const IterationState* state, bool first) {
        if (state->converged || state->failed) return;
        int id     = level.layout.active[blockIdx.x];
        float beta = first ? 0.0f : static_cast<float>(state->rz / state->previousRz);
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i        = id * blockCells + local;
            direction[i] = first ? level.solution[i] : level.solution[i] + beta * direction[i];
        }
    }

    __global__ void apply_matrix(PressureLevel level, const float* direction, float* product, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ double sums[threads];
        double sum = 0.0;
        int id     = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i        = id * blockCells + local;
            float result = matrix_product(level, direction, i);
            product[i]   = result;
            sum += static_cast<double>(direction[i]) * static_cast<double>(result);
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
            __syncthreads();
        }
        if (!threadIdx.x) {
            partial[blockIdx.x]                       = sums[0];
            partial[level.layout.blocks + blockIdx.x] = 0.0;
        }
    }

    __global__ void update_iterate(PressureLevel level, float* solution, float* residual, const float* direction, const float* product, double* partial, IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ double sums[threads];
        __shared__ float maxima[threads];
        double sum    = 0.0;
        float largest = 0.0f;
        int id        = level.layout.active[blockIdx.x];
        float alpha   = static_cast<float>(state->rz / state->denominator);
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            solution[i] += alpha * direction[i];
            float r     = residual[i] - alpha * product[i];
            residual[i] = r;
            sum += static_cast<double>(r) * static_cast<double>(r);
            largest = fmaxf(largest, fabsf(r));
        }
        sums[threadIdx.x]   = sum;
        maxima[threadIdx.x] = largest;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
                maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (!threadIdx.x) {
            partial[blockIdx.x]                       = sums[0];
            partial[level.layout.blocks + blockIdx.x] = maxima[0];
        }
        // Iteration count is advanced in a separate kernel to avoid a cross-CTA race.
    }

    __global__ void count_iteration(IterationState* state) {
        if (!state->converged && !state->failed) ++state->iterations;
    }

    __global__ void true_residual(PressureLevel level, const float* solution, float* residual) {
        int id = level.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i       = id * blockCells + local;
            residual[i] = level.rhs[i] - matrix_product(level, solution, i);
        }
    }

    __global__ void project_velocity(Grid grid, Fields fields, const float* pressure, float h, float3 wind) {
        int id = grid.layout.active[blockIdx.x];
        for (int local = threadIdx.x; local < blockCells; local += blockDim.x) {
            int i = id * blockCells + local;
            if (!grid.fluid[i]) {
                fields.smoke[i]       = 0.0f;
                fields.temperature[i] = 0.0f;
            }
            for (int axis = 0; axis < 3; ++axis) {
                int j      = neighbor(grid.layout, i, axis, -1);
                bool right = grid.fluid[i], left = j >= 0 && grid.fluid[j];
                if (grid.aperture[axis][i] <= 0.0f) fields.velocity[axis][i] = grid.solidVelocity[axis][i];
                else if (right || left) {
                    float difference = (right ? pressure[i] : 0.0f) - (left ? pressure[j] : 0.0f);
                    fields.velocity[axis][i] -= difference * (left && right ? 1.0f : 2.0f) / h;
                } else fields.velocity[axis][i] = component(wind, axis);
            }
        }
    }

    void precondition(PressureWorkspace workspace, cudaStream_t stream) {
        for (int l = 0; l < workspace.levelCount; ++l) {
            auto level       = workspace.levels[l];
            const float* rhs = l == 0 ? workspace.residual : level.rhs;
            zero_solution<<<level.layout.blocks, threads, 0, stream>>>(level, workspace.iteration);
            if (l + 1 == workspace.levelCount) solve_coarsest<<<1, threads, 0, stream>>>(level, rhs, workspace.iteration);
            else {
                smooth<<<level.layout.blocks, threads, 0, stream>>>(level, rhs, 0, workspace.iteration);
                smooth<<<level.layout.blocks, threads, 0, stream>>>(level, rhs, 1, workspace.iteration);
                restrict_residual<<<workspace.levels[l + 1].layout.blocks, threads, 0, stream>>>(level, workspace.levels[l + 1], rhs, workspace.iteration);
            }
        }
        for (int l = workspace.levelCount - 2; l >= 0; --l) {
            auto level       = workspace.levels[l];
            const float* rhs = l == 0 ? workspace.residual : level.rhs;
            prolongate<<<level.layout.blocks, threads, 0, stream>>>(level, workspace.levels[l + 1], workspace.iteration);
            smooth<<<level.layout.blocks, threads, 0, stream>>>(level, rhs, 1, workspace.iteration);
            smooth<<<level.layout.blocks, threads, 0, stream>>>(level, rhs, 0, workspace.iteration);
        }
    }

    void assemble_pressure(Grid grid, Fields fields, PressureWorkspace workspace, float h, bool obstacles, bool rebuild, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        int count = fine.layout.blocks;
        if (rebuild) assemble_fine<true><<<count, threads, 0, stream>>>(grid, fields, fine, h, workspace.parents, workspace.openComponents, workspace.anchors, workspace.solution, workspace.residual);
        else assemble_fine<false><<<count, threads, 0, stream>>>(grid, fields, fine, h, workspace.parents, workspace.openComponents, workspace.anchors, workspace.solution, workspace.residual);
        if (obstacles && rebuild) {
            connect_components<<<count, threads, 0, stream>>>(fine, grid.fluid, workspace.parents);
            mark_components<<<count, threads, 0, stream>>>(fine, workspace.parents, workspace.openComponents);
            anchor_components<<<count, threads, 0, stream>>>(fine, grid.fluid, workspace.parents, workspace.openComponents, workspace.anchors, workspace.residual);
            anchor_faces<<<count, threads, 0, stream>>>(fine, workspace.anchors);
        }
    }

    void coarse_topology(Layout fine, Layout coarse, cudaStream_t stream) {
        cuda_check(cudaMemsetAsync(coarse.keys, 255, coarse.tableCapacity * sizeof(unsigned long long), stream));
        cuda_check(cudaMemsetAsync(coarse.count, 0, sizeof(int), stream));
        coarse_keys<<<(fine.blocks + threads - 1) / threads, threads, 0, stream>>>(fine, coarse);
        coarse_indices<<<(coarse.tableCapacity + threads - 1) / threads, threads, 0, stream>>>(coarse);
    }

    void coarse_matrix(PressureLevel fine, PressureLevel coarse, cudaStream_t stream) {
        build_neighbors(coarse.layout, stream);
        coarsen_matrix<<<coarse.layout.blocks, threads, 0, stream>>>(fine, coarse);
    }

    void begin_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        int count = fine.layout.blocks;
        dot_blocks<<<count, threads, 0, stream>>>(fine, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, true);
        finish_dot<<<1, threads, 0, stream>>>(workspace.reductions, count, workspace.iteration, 2, config.pressureRelativeTolerance, config.divergenceTolerance * 0.25f);
        precondition(workspace, stream);
        dot_blocks<<<count, threads, 0, stream>>>(fine, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, false);
        finish_dot<<<1, threads, 0, stream>>>(workspace.reductions, count, workspace.iteration, 0, 0.0f, 0.0f);
        search_direction<<<count, threads, 0, stream>>>(fine, workspace.direction, workspace.iteration, true);
    }

    void iterate_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        int count = fine.layout.blocks;
        apply_matrix<<<count, threads, 0, stream>>>(fine, workspace.direction, workspace.product, workspace.reductions, workspace.iteration);
        finish_dot<<<1, threads, 0, stream>>>(workspace.reductions, count, workspace.iteration, 1, 0.0f, 0.0f);
        update_iterate<<<count, threads, 0, stream>>>(fine, workspace.solution, workspace.residual, workspace.direction, workspace.product, workspace.reductions, workspace.iteration);
        count_iteration<<<1, 1, 0, stream>>>(workspace.iteration);
        finish_dot<<<1, threads, 0, stream>>>(workspace.reductions, count, workspace.iteration, 3, config.pressureRelativeTolerance, config.divergenceTolerance * 0.25f);
        precondition(workspace, stream);
        dot_blocks<<<count, threads, 0, stream>>>(fine, workspace.residual, fine.solution, workspace.reductions, workspace.iteration, false);
        finish_dot<<<1, threads, 0, stream>>>(workspace.reductions, count, workspace.iteration, 0, 0.0f, 0.0f);
        search_direction<<<count, threads, 0, stream>>>(fine, workspace.direction, workspace.iteration, false);
    }

    void check_pressure(PressureWorkspace workspace, Config config, cudaStream_t stream) {
        auto fine = workspace.levels[0];
        true_residual<<<fine.layout.blocks, threads, 0, stream>>>(fine, workspace.solution, workspace.residual);
        dot_blocks<<<fine.layout.blocks, threads, 0, stream>>>(fine, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration, true);
        finish_dot<<<1, threads, 0, stream>>>(workspace.reductions, fine.layout.blocks, workspace.iteration, 3, config.pressureRelativeTolerance, config.divergenceTolerance * 0.5f);
    }

    void apply_pressure(Grid grid, Fields fields, const float* pressure, float h, float3 wind, cudaStream_t stream) {
        project_velocity<<<grid.layout.blocks, threads, 0, stream>>>(grid, fields, pressure, h, wind);
    }
} // namespace pyro::cuda
