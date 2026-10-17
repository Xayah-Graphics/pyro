#include "pressure.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>

namespace pyro::dense {
    enum struct DotOperation { update_rz, denominator, initial_norm, residual_norm, iteration_norm };

    template <typename T>
    __device__ T block_sum(T value) {
        __shared__ T partial[threads / 32];
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = value;
        __syncthreads();
        value = threadIdx.x < threads / 32 ? partial[threadIdx.x] : 0;
        if (threadIdx.x < 32)
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
        return value;
    }

    template <typename T>
    __device__ T block_maximum(T value) {
        __shared__ T partial[threads / 32];
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) value = max(value, __shfl_down_sync(0xffffffffu, value, offset));
        if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = value;
        __syncthreads();
        value = threadIdx.x < threads / 32 ? partial[threadIdx.x] : 0;
        if (threadIdx.x < 32)
#pragma unroll
            for (int offset = 16; offset; offset >>= 1) value = max(value, __shfl_down_sync(0xffffffffu, value, offset));
        return value;
    }

    template <bool regular, typename T, typename R = T>
    __device__ R matrix_product(PressureLevel level, const T* x, int i) {
        if (level.diagonal[i] <= 0.0f) return 0.0f;
        T center = x[i];
        R result = (regular ? .5f : 1.f) * level.boundary[i] * center;
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            int stride = axis == 0 ? 1 : axis == 1 ? level.layout.pitch : level.layout.slice;
            result += static_cast<R>((regular ? level.constant_coefficient : level.face[axis][i]) * (center - x[i - stride]));
            result += static_cast<R>((regular ? level.constant_coefficient : level.face[axis][i + stride]) * (center - x[i + stride]));
        }
        return result;
    }

    template <bool rebuild, BoundaryMode boundary_mode>
    __global__ void assemble_fine(Grid grid, Fields fields, PressureLevel level, float h, int* parents, int* open, unsigned char* anchors, double* solution, float* residual) {
        int id      = static_cast<int>(blockIdx.x);
        float scale = 1.0f / (h * h);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i          = id * tile_cells + local;
            float diagonal = 0.0f, boundary = 0.0f, divergence = boundary_mode == BoundaryMode::moving ? -grid.solid_divergence[i] : 0;
            bool is_open = false;
            if constexpr (rebuild && boundary_mode != BoundaryMode::none) {
                parents[i] = i;
                open[i]    = 0;
                anchors[i] = 0;
            }
            solution[i] = 0.0f;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                int previous = neighbor(grid.layout, i, axis, -1);
                if constexpr (rebuild) level.face[axis][i] = grid.fluid[i] && previous >= 0 && grid.fluid[previous] ? (boundary_mode == BoundaryMode::none ? scale : grid.aperture[axis][i] * scale) : 0.0f;
                for (int sign = -1; sign <= 1; sign += 2) {
                    int adjacent         = neighbor(grid.layout, i, axis, sign);
                    int face             = sign < 0 ? i : adjacent;
                    float aperture       = boundary_mode == BoundaryMode::none ? 1.f : value(grid.aperture[axis], face, 1.0f);
                    float fluid_velocity = value(fields.velocity[axis], face);
                    float solid_velocity = boundary_mode == BoundaryMode::moving ? value(grid.solid_velocity[axis], face) : 0;
                    divergence += sign * (aperture * fluid_velocity + (1.0f - aperture) * solid_velocity) / h;
                    if constexpr (rebuild) {
                        if (adjacent >= 0 && grid.fluid[adjacent]) diagonal += aperture * scale;
                        else if (adjacent < 0 || !interior(grid.layout, cell_coordinate(grid.layout, adjacent))) {
                            diagonal += 2.0f * aperture * scale;
                            boundary += 2.0f * aperture * scale;
                            is_open = is_open || aperture > 0.0f;
                        }
                    }
                }
            }
            if constexpr (rebuild) {
                level.diagonal[i] = grid.fluid[i] ? diagonal : 0.0f;
                level.boundary[i] = grid.fluid[i] ? boundary : 0.0f;
                if constexpr (boundary_mode != BoundaryMode::none) level.temporary[i] = is_open && grid.fluid[i] ? 1.0f : 0.0f;
            }
            level.rhs[i] = grid.fluid[i] && (boundary_mode == BoundaryMode::none || !anchors[i]) ? -divergence : 0.0f;
            residual[i]  = level.rhs[i];
            if constexpr (rebuild && boundary_mode != BoundaryMode::none) {
                int lane        = threadIdx.x & 31;
                unsigned starts = __ballot_sync(0xffffffffu, lane == 0 || level.face[0][i] <= 0);
                int root        = 31 - __clz(starts & (0xffffffffu >> (31 - lane)));
                parents[i]      = (i & ~31) + root;
            }
        }
    }

    __global__ void connect_components(PressureLevel level, const unsigned char* fluid, int* parents) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i    = blockIdx.x * tile_cells + local;
            int lane = threadIdx.x & 31;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                bool active   = fluid[i] && level.face[axis][i] > 0 && (axis != 0 || lane == 0);
                unsigned mask = __ballot_sync(0xffffffffu, active);
                if (!active) continue;
                int j = neighbor(level.layout, i, axis, -1);
                int a = atomicAdd(parents + i, 0), b = atomicAdd(parents + j, 0);
                unsigned long long key = (static_cast<unsigned long long>(static_cast<unsigned>(a)) << 32) | static_cast<unsigned>(b);
                unsigned peers         = __match_any_sync(mask, key);
                if (lane != __ffs(peers) - 1) continue;
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

    __global__ void mark_components(PressureLevel level, int* parents, int* open) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i          = blockIdx.x * tile_cells + local;
            int first      = atomicAdd(parents + i, 0);
            unsigned peers = __match_any_sync(0xffffffffu, first);
            int leader     = __ffs(peers) - 1;
            int r          = first;
            if ((threadIdx.x & 31) == leader) {
                int parent = atomicAdd(parents + r, 0);
                while (parent != r) {
                    int next = atomicAdd(parents + parent, 0);
                    atomicMin(parents + r, next);
                    r      = parent;
                    parent = next;
                }
            }
            r = __shfl_sync(0xffffffffu, r, leader);
            atomicExch(parents + i, r);
            unsigned marked = __ballot_sync(0xffffffffu, level.temporary[i] > 0);
            if ((threadIdx.x & 31) == leader && (marked & peers)) atomicExch(open + r, 1);
        }
    }

    __global__ void anchor_components(PressureLevel level, const unsigned char* fluid, int* parents, const int* open, unsigned char* anchors, float* residual) {
        int id = static_cast<int>(blockIdx.x);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i       = id * tile_cells + local;
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

    __global__ void anchor_faces(PressureLevel level, const unsigned char* anchors, int* indices, IterationState* state) {
        int id = static_cast<int>(blockIdx.x);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = id * tile_cells + local;
            if (anchors[i]) indices[atomicAdd(&state->iterations, 1)] = i;
#pragma unroll
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

    template <BoundaryMode boundary_mode>
    __global__ void closed_rhs(Grid grid, Fields fields, float h, PressureLevel level, PressureView workspace) {
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            double divergence = boundary_mode == BoundaryMode::moving ? -static_cast<double>(grid.solid_divergence[i]) : 0;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                int stride = axis == 0 ? 1 : axis == 1 ? grid.layout.pitch : grid.layout.slice;
                int right = i + stride;
                if (!grid.fluid[i]) break;
                double a = grid.aperture[axis][i], b = grid.aperture[axis][right];
                double flux = b * fields.velocity[axis][right] - a * fields.velocity[axis][i];
                if constexpr (boundary_mode == BoundaryMode::moving) flux += (1 - b) * grid.solid_velocity[axis][right] - (1 - a) * grid.solid_velocity[axis][i];
                divergence += flux / h;
            }
            workspace.rhs[i] = grid.fluid[i] ? -divergence : 0;
            double r = grid.fluid[i] && !workspace.anchors[i] ? -divergence : 0;
            workspace.closed_residual[i] = r;
            workspace.residual[i] = static_cast<float>(r);
            level.rhs[i] = static_cast<float>(r);
        }
    }

    __global__ void reference_residual(Grid grid, float h, PressureView workspace) {
        if (workspace.iteration->converged || workspace.iteration->failed) return;
        double largest = 0;
        for (int index = threadIdx.x; index < workspace.anchor_count; index += blockDim.x) {
            int i = workspace.anchor_indices[index];
            double r = workspace.rhs[i];
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                int stride = axis == 0 ? 1 : axis == 1 ? grid.layout.pitch : grid.layout.slice;
                int left = i - stride, right = i + stride;
                r -= static_cast<double>(grid.aperture[axis][i]) * (workspace.solution[i] - workspace.solution[left]) / (static_cast<double>(h) * h);
                r -= static_cast<double>(grid.aperture[axis][right]) * (workspace.solution[i] - workspace.solution[right]) / (static_cast<double>(h) * h);
            }
            largest = fmax(largest, fabs(r));
        }
        largest = block_maximum(largest);
        if (!threadIdx.x) workspace.reductions[grid.layout.blocks * 2] = largest;
    }

    __global__ void coarsen_matrix(PressureLevel fine, PressureLevel coarse) {
        int id = static_cast<int>(blockIdx.x);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i  = id * tile_cells + local;
            int3 c = cell_coordinate(coarse.layout, i);
            if (!interior(coarse.layout, c)) {
                coarse.diagonal[i] = coarse.boundary[i] = 0;
#pragma unroll
                for (int axis = 0; axis < 3; ++axis) coarse.face[axis][i] = 0;
                continue;
            }
            float diagonal = 0.0f, boundary = 0.0f;
            float faces[3]{};
            for (int z = 0; z < fine.coarsening.z; ++z)
                for (int y = 0; y < fine.coarsening.y; ++y)
                    for (int x = 0; x < fine.coarsening.x; ++x) {
                        int3 child{fine.coarsening.x * c.x + x, fine.coarsening.y * c.y + y, fine.coarsening.z * c.z + z};
                        if (!interior(fine.layout, child)) continue;
                        int j = cell_index(fine.layout, child);
                        if (j < 0) continue;
                        boundary += fine.boundary[j];
                        int parity[]{x, y, z};
#pragma unroll
                        for (int axis = 0; axis < 3; ++axis) {
                            int extent = axis == 0 ? fine.coarsening.x : axis == 1 ? fine.coarsening.y : fine.coarsening.z;
                            if (parity[axis] == extent - 1) diagonal += value(fine.face[axis], neighbor(fine.layout, j, axis, 1));
                            if (parity[axis] == 0) {
                                faces[axis] += fine.face[axis][j];
                                diagonal += fine.face[axis][j];
                            }
                        }
                    }
            coarse.diagonal[i] = .5f * (diagonal + boundary);
            coarse.boundary[i] = .5f * boundary;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) coarse.face[axis][i] = .5f * faces[axis];
        }
    }

    template <bool regular>
    __global__ void smooth_initial(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        constexpr float theta = 1.15f, delta = .85f;
        constexpr float rho = delta / theta, next_rho = 1 / (2 * theta / delta - rho);
        constexpr float alpha = 2 * next_rho / delta, beta = rho * next_rho;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i   = blockIdx.x * tile_cells + local;
            float d = level.diagonal[i];
            if (d <= 0) {
                level.solution[i] = 0;
                continue;
            }
            float x       = rhs[i] / (theta * d);
            float product = (regular ? .5f : 1.f) * level.boundary[i] * x;
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                int stride = axis == 0 ? 1 : axis == 1 ? level.layout.pitch : level.layout.slice;
                int a = i - stride, b = i + stride;
                float left  = level.diagonal[a] > 0 ? rhs[a] / (theta * level.diagonal[a]) : 0;
                float right = level.diagonal[b] > 0 ? rhs[b] / (theta * level.diagonal[b]) : 0;
                product += (regular ? level.constant_coefficient : level.face[axis][i]) * (x - left);
                product += (regular ? level.constant_coefficient : level.face[axis][b]) * (x - right);
            }
            level.solution[i] = (1 + beta) * x + alpha * (rhs[i] - product) / d;
        }
    }

    template <bool regular>
    __global__ void smooth_initial_tiled(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ float values[6][10][34];
        constexpr float theta = 1.15f, delta = .85f;
        constexpr float rho = delta / theta, next_rho = 1 / (2 * theta / delta - rho);
        constexpr float alpha = 2 * next_rho / delta, beta = rho * next_rho;
        int base_x = blockIdx.x * 32, base_y = blockIdx.y * 8, base_z = blockIdx.z * 4;
        for (int j = threadIdx.x; j < 6 * 10 * 34; j += blockDim.x) {
            int sx = j % 34, sy = j / 34 % 10, sz = j / 340;
            int x = base_x + sx - 1, y = base_y + sy - 1, z = base_z + sz - 1;
            int i       = x + y * level.layout.pitch + z * level.layout.slice;
            float value = 0;
            if (x >= 0 && x < level.layout.pitch && y >= 0 && y < level.layout.size.y + 2 && z >= 0 && i < level.layout.cells) {
                float d = level.diagonal[i];
                if (d > 0) value = rhs[i] / (theta * d);
            }
            values[sz][sy][sx] = value;
        }
        __syncthreads();
        int x = base_x + (threadIdx.x & 31), y = base_y + threadIdx.x / 32;
        if (x >= level.layout.pitch || y >= level.layout.size.y + 2) return;
        for (int z = 0; z < 4; ++z) {
            int i = x + y * level.layout.pitch + (base_z + z) * level.layout.slice;
            if (i >= level.layout.cells) continue;
            float d      = level.diagonal[i];
            float result = 0;
            if (d > 0) {
                int sx = (threadIdx.x & 31) + 1, sy = threadIdx.x / 32 + 1, sz = z + 1;
                float v       = values[sz][sy][sx];
                float product = (regular ? .5f : 1.f) * level.boundary[i] * v;
                product += (regular ? level.constant_coefficient : level.face[0][i]) * (v - values[sz][sy][sx - 1]);
                product += (regular ? level.constant_coefficient : level.face[0][i + 1]) * (v - values[sz][sy][sx + 1]);
                product += (regular ? level.constant_coefficient : level.face[1][i]) * (v - values[sz][sy - 1][sx]);
                product += (regular ? level.constant_coefficient : level.face[1][i + level.layout.pitch]) * (v - values[sz][sy + 1][sx]);
                product += (regular ? level.constant_coefficient : level.face[2][i]) * (v - values[sz - 1][sy][sx]);
                product += (regular ? level.constant_coefficient : level.face[2][i + level.layout.slice]) * (v - values[sz + 1][sy][sx]);
                result = (1 + beta) * v + alpha * (rhs[i] - product) / d;
            }
            level.solution[i] = result;
        }
    }

    template <bool regular>
    __global__ void chebyshev_first(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        constexpr float theta = 1.15f;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i              = blockIdx.x * tile_cells + local;
            float d            = level.diagonal[i];
            float x            = level.solution[i];
            float r            = rhs[i] - matrix_product<regular>(level, level.solution, i);
            level.temporary[i] = d > 0 ? x + r / (theta * d) : 0;
        }
    }

    template <bool dot, bool regular>
    __global__ void chebyshev_second(PressureLevel level, const float* rhs, const IterationState* state, double* partial) {
        if (state->converged || state->failed) return;
        constexpr float theta = 1.15f, delta = .85f;
        constexpr float rho = delta / theta, next_rho = 1 / (2 * theta / delta - rho);
        constexpr float alpha = 2 * next_rho / delta, beta = rho * next_rho;
        double sum = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i             = blockIdx.x * tile_cells + local;
            float d           = level.diagonal[i];
            float old         = level.solution[i];
            float x           = level.temporary[i];
            float r           = rhs[i] - matrix_product<regular>(level, level.temporary, i);
            float result      = d > 0 ? x + beta * (x - old) + alpha * r / d : 0;
            level.solution[i] = result;
            if constexpr (dot) sum += static_cast<double>(rhs[i]) * result;
        }
        if constexpr (dot) {
            sum = block_sum(sum);
            if (!threadIdx.x) partial[blockIdx.x] = sum;
        }
    }

    template <bool regular>
    __global__ void prepare_restriction(PressureLevel fine, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i             = blockIdx.x * tile_cells + local;
            fine.temporary[i] = fine.diagonal[i] > 0 ? (rhs[i] - matrix_product<regular>(fine, fine.solution, i)) / fine.diagonal[i] : 0;
        }
    }

    template <bool regular>
    __global__ void restrict_residual(PressureLevel fine, PressureLevel coarse, const IterationState* state) {
        if (state->converged || state->failed) return;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i     = blockIdx.x * tile_cells + local;
            int3 c    = cell_coordinate(coarse.layout, i);
            float sum = 0;
            if (interior(coarse.layout, c))
                for (int z = 0; z < fine.coarsening.z; ++z)
                    for (int y = 0; y < fine.coarsening.y; ++y)
                        for (int x = 0; x < fine.coarsening.x; ++x) {
                            int3 p{fine.coarsening.x * c.x + x, fine.coarsening.y * c.y + y, fine.coarsening.z * c.z + z};
                            if (!interior(fine.layout, p)) continue;
                            int j = cell_index(fine.layout, p);
                            if (fine.diagonal[j] > 0) sum += fine.diagonal[j] * fine.temporary[j] - (2.f / 3.f) * matrix_product<regular>(fine, fine.temporary, j);
                        }
            coarse.rhs[i] = sum;
        }
    }

    __global__ void inject_correction(PressureLevel fine, PressureLevel coarse, const IterationState* state) {
        if (state->converged || state->failed) return;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i             = blockIdx.x * tile_cells + local;
            int3 p            = cell_coordinate(fine.layout, i);
            fine.temporary[i] = fine.diagonal[i] > 0 ? value(coarse.solution, cell_index(coarse.layout, {p.x / fine.coarsening.x, p.y / fine.coarsening.y, p.z / fine.coarsening.z})) : 0;
        }
    }

    template <bool regular>
    __global__ void prolongate(PressureLevel fine, const IterationState* state) {
        if (state->converged || state->failed) return;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            if (fine.diagonal[i] > 0) fine.solution[i] += fine.temporary[i] - (2.f / 3.f) * matrix_product<regular>(fine, fine.temporary, i) / fine.diagonal[i];
        }
    }

    __global__ void solve_coarsest(PressureLevel level, const float* rhs, const IterationState* state) {
        if (state->converged || state->failed) return;
        __shared__ float x[512], b[512], diagonal[512], boundary[512], face[3][512];
        int nx = level.layout.size.x, ny = level.layout.size.y, nz = level.layout.size.z;
        int count = nx * ny * nz;
        for (int j = threadIdx.x; j < count; j += blockDim.x) {
            int z = j / (nx * ny), y = j / nx % ny, px = j % nx;
            int i       = px + 1 + (y + 1) * level.layout.pitch + (z + 1) * level.layout.slice;
            x[j]        = 0;
            b[j]        = rhs[i];
            diagonal[j] = level.diagonal[i];
            boundary[j] = level.boundary[i];
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) face[axis][j] = level.face[axis][i];
        }
        __syncthreads();
        for (int sweep = 0; sweep < 8; ++sweep)
            for (int phase = 0; phase < 4; ++phase) {
                int color = phase == 0 || phase == 3 ? 0 : 1;
                for (int j = threadIdx.x; j < count; j += blockDim.x) {
                    int c[3]{j % nx, j / nx % ny, j / (nx * ny)};
                    if (((c[0] + c[1] + c[2]) & 1) != color || diagonal[j] <= 0) continue;
                    float product = boundary[j] * x[j];
#pragma unroll
                    for (int axis = 0; axis < 3; ++axis) {
                        int stride = axis == 0 ? 1 : axis == 1 ? nx : nx * ny;
                        int size   = axis == 0 ? nx : axis == 1 ? ny : nz;
                        if (c[axis] > 0) product += face[axis][j] * (x[j] - x[j - stride]);
                        if (c[axis] + 1 < size) product += face[axis][j + stride] * (x[j] - x[j + stride]);
                    }
                    x[j] += (b[j] - product) / diagonal[j];
                }
                __syncthreads();
            }
        for (int j = threadIdx.x; j < count; j += blockDim.x) {
            int z = j / (nx * ny), y = j / nx % ny, px = j % nx;
            int i             = px + 1 + (y + 1) * level.layout.pitch + (z + 1) * level.layout.slice;
            level.solution[i] = x[j];
        }
    }

    template <DotOperation operation, bool closed = false>
    __global__ void finish_dot(double* partial, int count, IterationState* state, float relative_tolerance, float absolute_tolerance) {
        if (state->converged || state->failed) return;
        constexpr bool maximum = operation == DotOperation::initial_norm || operation == DotOperation::residual_norm || operation == DotOperation::iteration_norm;
        double sum = 0, largest = 0;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            sum += partial[i];
            if constexpr (maximum) largest = fmax(largest, partial[count + i]);
        }
        sum = block_sum(sum);
        if constexpr (maximum) largest = block_maximum(largest);
        if (threadIdx.x) return;
        if constexpr (maximum && closed) largest = fmax(largest, partial[count * 2]);
        if constexpr (operation == DotOperation::iteration_norm) ++state->iterations;
        if (!isfinite(sum)) {
            state->failed = 1;
            return;
        }
        if constexpr (operation == DotOperation::update_rz) {
            state->previous_rz = state->rz;
            state->rz          = sum;
        } else if constexpr (operation == DotOperation::denominator) {
            state->denominator = sum;
            if (sum <= 0) state->failed = 1;
        } else {
            if constexpr (operation == DotOperation::initial_norm) state->initial_norm = sum;
            state->max_residual = static_cast<float>(largest);
            bool relative       = sum <= state->initial_norm * relative_tolerance * relative_tolerance;
            state->converged    = largest <= absolute_tolerance && (relative || largest <= absolute_tolerance * .1f);
            if constexpr (operation == DotOperation::initial_norm) {
                if (state->iterations > 0) {
                    state->converged = 0;
                    if (sum == 0) state->failed = 1;
                }
            }
        }
    }

    template <bool maximum>
    __global__ void dot_blocks(PressureLevel level, const float* a, const float* b, double* partial, const IterationState* state) {
        if (state->converged || state->failed) return;
        double sum    = 0;
        float largest = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = blockIdx.x * tile_cells + local;
            sum += static_cast<double>(a[i]) * b[i];
            if constexpr (maximum) largest = fmaxf(largest, fabsf(a[i]));
        }
        sum = block_sum(sum);
        if constexpr (maximum) largest = block_maximum(largest);
        if (!threadIdx.x) {
            partial[blockIdx.x] = sum;
            if constexpr (maximum) partial[level.layout.blocks + blockIdx.x] = largest;
        }
    }

    template <bool first, bool closed = false>
    __global__ void search_direction(PressureLevel level, float* direction, const IterationState* state, int maximum, cudaGraphConditionalHandle condition, const unsigned char* anchors) {
        if constexpr (!first)
            if (!blockIdx.x && !threadIdx.x) cudaGraphSetConditional(condition, !state->converged && !state->failed && state->iterations < maximum);
        if (state->converged || state->failed) return;
        int id     = static_cast<int>(blockIdx.x);
        float beta = first ? 0.0f : static_cast<float>(state->rz / state->previous_rz);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i        = id * tile_cells + local;
            direction[i] = first ? level.solution[i] : level.solution[i] + beta * direction[i];
            if constexpr (closed) if (anchors[i]) direction[i] = 0;
        }
    }

    template <bool regular, bool closed = false>
    __global__ void apply_matrix(PressureLevel level, const float* direction, float* product, double* partial, const IterationState* state, double* closed_product) {
        if (state->converged || state->failed) return;
        double sum = 0.0;
        int id     = static_cast<int>(blockIdx.x);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i        = id * tile_cells + local;
            auto result = matrix_product<regular, float, std::conditional_t<closed, double, float>>(level, direction, i);
            if constexpr (closed) closed_product[i] = result;
            else product[i] = result;
            sum += static_cast<double>(direction[i]) * static_cast<double>(result);
        }
        sum = block_sum(sum);
        if (!threadIdx.x) partial[blockIdx.x] = sum;
    }

    template <bool closed>
    __global__ void update_iterate(PressureLevel level, double* solution, float* residual, const float* direction, const float* product, double* partial, IterationState* state, double* closed_residual, const double* closed_product) {
        if (state->converged || state->failed) return;
        double sum    = 0.0;
        float largest = 0.0f;
        int id        = static_cast<int>(blockIdx.x);
        std::conditional_t<closed, double, float> alpha = static_cast<std::conditional_t<closed, double, float>>(state->rz / state->denominator);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = id * tile_cells + local;
            float r;
            if constexpr (closed) {
                solution[i] += alpha * direction[i];
                double next = closed_residual[i] - alpha * closed_product[i];
                closed_residual[i] = next;
                r = static_cast<float>(next);
            } else {
                solution[i] += static_cast<double>(alpha) * direction[i];
                r = residual[i] - alpha * product[i];
            }
            residual[i] = r;
            sum += static_cast<double>(r) * static_cast<double>(r);
            largest = fmaxf(largest, fabsf(r));
        }
        sum     = block_sum(sum);
        largest = block_maximum(largest);
        if (!threadIdx.x) {
            partial[blockIdx.x]                       = sum;
            partial[level.layout.blocks + blockIdx.x] = largest;
        }
    }

    template <BoundaryMode boundary_mode, bool closed>
    __global__ void true_residual(PressureLevel level, Grid grid, Fields fields, float h, const unsigned char* anchors, const double* solution, float* residual, double* partial, const double* rhs, double* closed_residual) {
        double sum    = 0;
        float largest = 0;
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i       = blockIdx.x * tile_cells + local;
            double exact = (closed ? (anchors[i] ? 0 : rhs[i]) : level.rhs[i]) - matrix_product<boundary_mode == BoundaryMode::none>(level, solution, i);
            if constexpr (closed) closed_residual[i] = exact;
            float r = static_cast<float>(exact);
            residual[i] = r;
            if constexpr (boundary_mode != BoundaryMode::none) {
                if (anchors[i]) {
                    double divergence = boundary_mode == BoundaryMode::moving ? -grid.solid_divergence[i] : 0;
#pragma unroll
                    for (int axis = 0; axis < 3; ++axis) {
                        int stride = axis == 0 ? 1 : axis == 1 ? grid.layout.pitch : grid.layout.slice;
#pragma unroll
                        for (int sign = -1; sign <= 1; sign += 2) {
                            int face         = sign < 0 ? i : i + stride;
                            int left         = face - stride;
                            bool right_fluid = grid.fluid[face], left_fluid = grid.fluid[left];
                            float aperture       = grid.aperture[axis][face];
                            float velocity       = fields.velocity[axis][face];
                            float solid_velocity = boundary_mode == BoundaryMode::moving ? grid.solid_velocity[axis][face] : 0;
                            if (aperture <= 0) velocity = solid_velocity;
                            else if (right_fluid || left_fluid) {
                                float difference = (right_fluid ? solution[face] : 0) - (left_fluid ? solution[left] : 0);
                                velocity -= difference * (right_fluid && left_fluid ? 1.f : 2.f) / h;
                            }
                            divergence += sign * (static_cast<double>(aperture) * velocity + (1.0 - aperture) * solid_velocity) / h;
                        }
                    }
                    r = -divergence;
                }
            }
            sum += static_cast<double>(r) * r;
            largest = fmaxf(largest, fabsf(r));
        }
        sum     = block_sum(sum);
        largest = block_maximum(largest);
        if (!threadIdx.x) {
            partial[blockIdx.x]                       = sum;
            partial[level.layout.blocks + blockIdx.x] = largest;
        }
    }

    template <BoundaryMode boundary_mode>
    __global__ void project_velocity(Grid grid, Fields fields, const double* pressure, float h, float3 wind) {
        int id = static_cast<int>(blockIdx.x);
        for (int local = threadIdx.x; local < tile_cells; local += blockDim.x) {
            int i = id * tile_cells + local;
            if (!grid.fluid[i]) {
                fields.smoke[i]       = 0.0f;
                fields.temperature[i] = 0.0f;
            }
#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                if (!face_valid(grid.layout, cell_coordinate(grid.layout, i), axis)) {
                    fields.velocity[axis][i] = (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z);
                    continue;
                }
                int j      = neighbor(grid.layout, i, axis, -1);
                bool right = grid.fluid[i], left = j >= 0 && grid.fluid[j];
                if (boundary_mode != BoundaryMode::none && grid.aperture[axis][i] <= 0.0f) fields.velocity[axis][i] = boundary_mode == BoundaryMode::moving ? grid.solid_velocity[axis][i] : 0;
                else if (right || left) {
                    float difference = (right ? pressure[i] : 0.0f) - (left ? pressure[j] : 0.0f);
                    fields.velocity[axis][i] -= difference * (left && right ? 1.0f : 2.0f) / h;
                } else fields.velocity[axis][i] = (axis == 0 ? wind.x : axis == 1 ? wind.y : wind.z);
            }
        }
    }

    void precondition(std::span<const PressureLevel> levels, PressureView workspace, ::cuda::stream_ref stream) {
        for (int l = 0; l < workspace.level_count; ++l) {
            auto level       = levels[l];
            const float* rhs = l == 0 ? workspace.residual : level.rhs;
            if (l + 1 == workspace.level_count) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), solve_coarsest, level, rhs, workspace.iteration);
            else {
                auto launch = [&]<bool regular>() {
                    if (level.layout.size.x >= 64) {
                        int planes = (level.layout.cells + level.layout.slice - 1) / level.layout.slice;
                        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(dim3((level.layout.pitch + 31) / 32, (level.layout.size.y + 9) / 8, (planes + 3) / 4)), ::cuda::block_dims<256>()), smooth_initial_tiled<regular>, level, rhs, workspace.iteration);
                    } else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<threads>()), smooth_initial<regular>, level, rhs, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<threads>()), prepare_restriction<regular>, level, rhs, workspace.iteration);
                    ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(levels[l + 1].layout.blocks), ::cuda::block_dims<threads>()), restrict_residual<regular>, level, levels[l + 1], workspace.iteration);
                };
                if (level.constant_coefficient > 0) launch.template operator()<true>();
                else launch.template operator()<false>();
            }
        }
        for (int l = workspace.level_count - 2; l >= 0; --l) {
            auto level       = levels[l];
            const float* rhs = l == 0 ? workspace.residual : level.rhs;
            auto launch      = [&]<bool regular>() {
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<threads>()), inject_correction, level, levels[l + 1], workspace.iteration);
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<threads>()), prolongate<regular>, level, workspace.iteration);
                ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<threads>()), chebyshev_first<regular>, level, rhs, workspace.iteration);
                if (l == 0) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<threads>()), chebyshev_second<true, regular>, level, rhs, workspace.iteration, workspace.reductions);
                else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(level.layout.blocks), ::cuda::block_dims<threads>()), chebyshev_second<false, regular>, level, rhs, workspace.iteration, workspace.reductions);
            };
            if (level.constant_coefficient > 0) launch.template operator()<true>();
            else launch.template operator()<false>();
        }
    }

    void assemble_pressure(std::span<const PressureLevel> levels, Grid grid, Fields fields, PressureView& workspace, float h, bool rebuild, ::cuda::stream_ref stream) {
        auto fine   = levels[0];
        int count   = fine.layout.blocks;
        int* open_components = workspace.anchor_indices;
        auto launch = [&]<BoundaryMode boundary_mode>() {
            if (rebuild) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), assemble_fine<true, boundary_mode>, grid, fields, fine, h, workspace.parents, open_components, workspace.anchors, workspace.solution, workspace.residual);
            else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), assemble_fine<false, boundary_mode>, grid, fields, fine, h, workspace.parents, open_components, workspace.anchors, workspace.solution, workspace.residual);
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none: launch.template operator()<BoundaryMode::none>(); break;
        case BoundaryMode::stationary: launch.template operator()<BoundaryMode::stationary>(); break;
        case BoundaryMode::moving: launch.template operator()<BoundaryMode::moving>(); break;
        }
        if (rebuild) workspace.anchor_count = 0;
        if (grid.boundary_mode != BoundaryMode::none && rebuild) {
            ::cuda::fill_bytes(stream, ::cuda::std::span(workspace.iteration, 1), 0);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), connect_components, fine, grid.fluid, workspace.parents);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), mark_components, fine, workspace.parents, open_components);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), anchor_components, fine, grid.fluid, workspace.parents, open_components, workspace.anchors, workspace.residual);
            ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), anchor_faces, fine, workspace.anchors, workspace.anchor_indices, workspace.iteration);
            IterationState status{};
            ::cuda::copy_bytes(stream, ::cuda::std::span(workspace.iteration, 1), ::cuda::std::span(&status, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            workspace.anchor_count = status.iterations;
        }
    }

    void coarse_matrix(PressureLevel fine, PressureLevel coarse, ::cuda::stream_ref stream) {
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(coarse.layout.blocks), ::cuda::block_dims<threads>()), coarsen_matrix, fine, coarse);
    }

    template <bool closed>
    void begin_pressure(std::span<const PressureLevel> levels, Grid grid, float h, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream) {
        auto fine = levels[0];
        int count = fine.layout.blocks;
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), dot_blocks<true>, fine, workspace.residual, workspace.residual, workspace.reductions, workspace.iteration);
        if constexpr (closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), reference_residual, grid, h, workspace);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_dot<DotOperation::initial_norm, closed>, workspace.reductions, count, workspace.iteration, config.relative_tolerance, config.divergence_tolerance * 0.25f);
        precondition(levels, workspace, stream);
        if (workspace.level_count == 1) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), dot_blocks<false>, fine, workspace.residual, fine.solution, workspace.reductions, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_dot<DotOperation::update_rz>, workspace.reductions, count, workspace.iteration, 0.0f, 0.0f);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), search_direction<true, closed>, fine, workspace.direction, workspace.iteration, config.max_iterations, cudaGraphConditionalHandle{}, workspace.anchors);
    }

    template <bool closed>
    void iterate_pressure(std::span<const PressureLevel> levels, Grid grid, float h, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream, cudaGraphConditionalHandle condition) {
        auto fine = levels[0];
        int count = fine.layout.blocks;
        if constexpr (closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), apply_matrix<false, true>, fine, workspace.direction, workspace.product, workspace.reductions, workspace.iteration, workspace.closed_product);
        else if (fine.constant_coefficient > 0) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), apply_matrix<true>, fine, workspace.direction, workspace.product, workspace.reductions, workspace.iteration, nullptr);
        else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), apply_matrix<false>, fine, workspace.direction, workspace.product, workspace.reductions, workspace.iteration, nullptr);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_dot<DotOperation::denominator>, workspace.reductions, count, workspace.iteration, 0.0f, 0.0f);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), update_iterate<closed>, fine, workspace.solution, workspace.residual, workspace.direction, workspace.product, workspace.reductions, workspace.iteration, workspace.closed_residual, workspace.closed_product);
        if constexpr (closed) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), reference_residual, grid, h, workspace);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_dot<DotOperation::iteration_norm, closed>, workspace.reductions, count, workspace.iteration, config.relative_tolerance, config.divergence_tolerance * 0.25f);
        precondition(levels, workspace, stream);
        if (workspace.level_count == 1) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), dot_blocks<false>, fine, workspace.residual, fine.solution, workspace.reductions, workspace.iteration);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_dot<DotOperation::update_rz>, workspace.reductions, count, workspace.iteration, 0.0f, 0.0f);
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(count), ::cuda::block_dims<threads>()), search_direction<false, closed>, fine, workspace.direction, workspace.iteration, config.max_iterations, condition, workspace.anchors);
    }

    void check_pressure(std::span<const PressureLevel> levels, Grid grid, Fields fields, float h, PressureView workspace, PressureConfig config, ::cuda::stream_ref stream) {
        auto fine   = levels[0];
        auto launch = [&]<BoundaryMode boundary_mode>() {
            if (workspace.anchor_count) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<threads>()), true_residual<boundary_mode, true>, fine, grid, fields, h, workspace.anchors, workspace.solution, workspace.residual, workspace.reductions, workspace.rhs, workspace.closed_residual);
            else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(fine.layout.blocks), ::cuda::block_dims<threads>()), true_residual<boundary_mode, false>, fine, grid, fields, h, workspace.anchors, workspace.solution, workspace.residual, workspace.reductions, nullptr, nullptr);
        };
        switch (grid.boundary_mode) {
        case BoundaryMode::none: launch.template operator()<BoundaryMode::none>(); break;
        case BoundaryMode::stationary: launch.template operator()<BoundaryMode::stationary>(); break;
        case BoundaryMode::moving: launch.template operator()<BoundaryMode::moving>(); break;
        }
        ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims<1>(), ::cuda::block_dims<threads>()), finish_dot<DotOperation::residual_norm>, workspace.reductions, fine.layout.blocks, workspace.iteration, config.relative_tolerance, config.divergence_tolerance * 0.5f);
    }

    void apply_pressure(Grid grid, Fields fields, const double* pressure, float h, float3 wind, ::cuda::stream_ref stream) {
        switch (grid.boundary_mode) {
        case BoundaryMode::none: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), project_velocity<BoundaryMode::none>, grid, fields, pressure, h, wind); break;
        case BoundaryMode::stationary: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), project_velocity<BoundaryMode::stationary>, grid, fields, pressure, h, wind); break;
        case BoundaryMode::moving: ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), project_velocity<BoundaryMode::moving>, grid, fields, pressure, h, wind); break;
        }
    }

    Pressure::Pressure(int3 resolution, PressureConfig settings, ::cuda::stream_ref stream, ::cuda::device_memory_pool_ref pool)
        : config(settings),
          stream(stream),
          iteration_graph(nullptr, cudaGraphExecDestroy),
          iteration_storage(::cuda::make_buffer<IterationState>(stream, pool, 1, ::cuda::no_init)),
          index_storage(stream, pool),
          reduction_storage(stream, pool),
          closed_storage(stream, pool) {
        view.iteration = iteration_storage.data();
        ::cuda::fill_bytes(stream, iteration_storage, 0);
        hierarchy.resize(std::bit_width(static_cast<unsigned>(std::max({resolution.x, resolution.y, resolution.z}))));
        int3 size = resolution;
        for (auto& level : hierarchy) {
            auto& layout     = level.layout;
            layout.size      = size;
            layout.pitch     = size.x + 2;
            layout.slice     = layout.pitch * (size.y + 2);
            layout.cells     = (layout.slice * (size.z + 2) + tile_cells - 1) / tile_cells * tile_cells;
            layout.blocks    = layout.cells / tile_cells;
            level.coarsening = {size.x > 1 ? 2 : 1, size.y > 1 ? 2 : 1, size.z > 1 ? 2 : 1};
            auto cells       = static_cast<std::size_t>(layout.cells);
            auto& storage    = level_storage.emplace_back(::cuda::make_buffer<float>(stream, pool, cells * 8, ::cuda::no_init));
            level.diagonal   = storage.data();
            level.boundary   = storage.data() + cells;
            for (int axis = 0; axis < 3; ++axis) level.face[axis] = storage.data() + cells * (axis + 2);
            level.solution  = storage.data() + cells * 5;
            level.rhs       = storage.data() + cells * 6;
            level.temporary = storage.data() + cells * 7;
            if (view.level_count == 0) {
                ::cuda::fill_bytes(stream, storage.first(cells * 5), 0);
                ::cuda::fill_bytes(stream, storage.subspan(cells * 6), 0);
            } else ::cuda::fill_bytes(stream, storage, 0);
            ++view.level_count;
            if (size.x * size.y * size.z <= 512) break;
            size = {(size.x + 1) / 2, (size.y + 1) / 2, (size.z + 1) / 2};
        }
        auto layout       = hierarchy[0].layout;
        auto cells        = static_cast<std::size_t>(layout.cells);
        index_storage     = ::cuda::make_buffer<int>(stream, pool, cells * 2, ::cuda::no_init);
        reduction_storage = ::cuda::make_buffer<double>(stream, pool, cells + layout.blocks * 6ull + 6, ::cuda::no_init);
        ::cuda::fill_bytes(stream, index_storage, 0);
        ::cuda::fill_bytes(stream, reduction_storage, 0);
        view.parents         = index_storage.data();
        view.anchor_indices = index_storage.data() + cells;
        view.solution        = reduction_storage.data();
        view.reductions      = reduction_storage.data() + cells;
    }

    void Pressure::build(Grid grid, float h) {
        float coefficient = grid.boundary_mode == BoundaryMode::none ? 1.f / (h * h) : 0;
        if (hierarchy[0].constant_coefficient != coefficient) iteration_graph.reset();
        for (int l = 0; l < view.level_count; ++l) {
            auto& level                = hierarchy[l];
            level.constant_coefficient = coefficient;
            auto n                     = level.layout.size;
            coefficient                = (n.x % 2 || n.y % 2 || n.z % 2 || n.x <= 1 || n.y <= 1 || n.z <= 1) ? 0 : coefficient * 2;
        }
        auto levels = std::span(hierarchy).first(view.level_count);
        int previous_anchors = view.anchor_count;
        assemble_pressure(levels, grid, grid.state, view, h, true, stream);
        if (previous_anchors != view.anchor_count) iteration_graph.reset();
        for (int l = 1; l < view.level_count; ++l) coarse_matrix(levels[l - 1], levels[l], stream);
        matrix_dirty = false;
    }

    int Pressure::project(Grid grid, Fields fields, float h, float3 wind) {
        float coefficient = grid.boundary_mode == BoundaryMode::none ? 1.f / (h * h) : 0;
        if (hierarchy[0].constant_coefficient != coefficient) iteration_graph.reset();
        for (int l = 0; l < view.level_count; ++l) {
            auto& level                = hierarchy[l];
            level.constant_coefficient = coefficient;
            auto n                     = level.layout.size;
            coefficient                = (n.x % 2 || n.y % 2 || n.z % 2 || n.x <= 1 || n.y <= 1 || n.z <= 1) ? 0 : coefficient * 2;
        }
        auto levels = std::span(hierarchy).first(view.level_count);

        auto& workspace = view;
        int previous_anchors = workspace.anchor_count;
        assemble_pressure(levels, grid, fields, workspace, h, matrix_dirty, stream);
        if (previous_anchors != workspace.anchor_count) iteration_graph.reset();
        if (matrix_dirty)
            for (int l = 1; l < workspace.level_count; ++l) coarse_matrix(levels[l - 1], levels[l], stream);
        matrix_dirty = false;
        if (workspace.anchor_count) {
            auto cells = static_cast<std::size_t>(grid.layout.cells);
            if (closed_storage.empty()) {
                closed_storage = ::cuda::make_buffer<double>(stream, reduction_storage.memory_resource(), cells * 3, ::cuda::no_init);
                workspace.rhs = closed_storage.data();
                workspace.closed_residual = closed_storage.data() + cells;
                workspace.closed_product = closed_storage.data() + cells * 2;
            }
            if (grid.boundary_mode == BoundaryMode::moving) ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), closed_rhs<BoundaryMode::moving>, grid, fields, h, levels[0], workspace);
            else ::cuda::launch(stream, ::cuda::make_config(::cuda::grid_dims(grid.layout.blocks), ::cuda::block_dims<threads>()), closed_rhs<BoundaryMode::stationary>, grid, fields, h, levels[0], workspace);
        }
        ::cuda::fill_bytes(stream, ::cuda::std::span(workspace.iteration, 1), 0);
        if (workspace.anchor_count) begin_pressure<true>(levels, grid, h, workspace, config, stream);
        else begin_pressure<false>(levels, grid, h, workspace, config, stream);
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
            if (workspace.anchor_count) iterate_pressure<true>(levels, grid, h, workspace, config, stream, condition);
            else iterate_pressure<false>(levels, grid, h, workspace, config, stream, condition);
            if (auto result = cudaStreamEndCapture(stream.get(), &body); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            cudaGraphExec_t executable{};
            if (auto result = cudaGraphInstantiate(&executable, graph.get(), 0); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            iteration_graph.reset(executable);
        }
        IterationState status{};
        for (;;) {
            if (auto result = cudaGraphLaunch(iteration_graph.get(), stream.get()); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
            ::cuda::copy_bytes(stream, ::cuda::std::span(workspace.iteration, 1), ::cuda::std::span(&status, 1), {.src_access_order = ::cuda::source_access_order::stream});
            stream.sync();
            if (status.failed) break;
            if (status.converged || status.iterations >= config.max_iterations) {
                status.converged = 0;
                ::cuda::copy_bytes(stream, ::cuda::std::span(&status, 1), ::cuda::std::span(workspace.iteration, 1), {.src_access_order = ::cuda::source_access_order::stream});
                check_pressure(levels, grid, fields, h, workspace, config, stream);
                ::cuda::copy_bytes(stream, ::cuda::std::span(workspace.iteration, 1), ::cuda::std::span(&status, 1), {.src_access_order = ::cuda::source_access_order::stream});
                stream.sync();
                if (status.max_residual <= config.divergence_tolerance * 0.5f || status.failed || status.iterations >= config.max_iterations) break;
                status.converged = 0;
                ::cuda::copy_bytes(stream, ::cuda::std::span(&status, 1), ::cuda::std::span(workspace.iteration, 1), {.src_access_order = ::cuda::source_access_order::stream});
                if (workspace.anchor_count) begin_pressure<true>(levels, grid, h, workspace, config, stream);
                else begin_pressure<false>(levels, grid, h, workspace, config, stream);
            }
        }
        if (status.failed || status.max_residual > config.divergence_tolerance * 0.5f) throw std::runtime_error("Pressure solve failed: iterations=" + std::to_string(status.iterations) + ", true residual=" + std::to_string(status.max_residual));
        apply_pressure(grid, fields, workspace.solution, h, wind, stream);
        return status.iterations;
    }

}
