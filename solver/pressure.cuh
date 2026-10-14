#ifndef PYRO_PRESSURE_CUH
#define PYRO_PRESSURE_CUH
#include <cuda_runtime.h>

namespace pyro::cuda {
    struct IterationState {
        double rz{}, previousRz{}, denominator{}, initialNorm{};
        float maxResidual{};
        int iterations{}, converged{}, failed{};
    };

#ifdef __CUDACC__
    __device__ inline int component_root(int* parents, int i) {
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

    template <int threadCount>
    __global__ void finish_dot(double* partial, int count, IterationState* state, int operation, float relativeTolerance, float absoluteTolerance) {
        if (state->converged || state->failed) return;
        __shared__ double sums[threadCount], maxima[threadCount];
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
            state->maxResidual = static_cast<float>(maxima[0]);
            bool relative      = sums[0] <= state->initialNorm * relativeTolerance * relativeTolerance;
            state->converged   = maxima[0] <= absoluteTolerance && (relative || maxima[0] <= absoluteTolerance * 0.1f);
        }
    }
#endif
} // namespace pyro::cuda
#endif
