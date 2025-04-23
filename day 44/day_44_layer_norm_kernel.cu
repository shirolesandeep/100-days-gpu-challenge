#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

// Half-precision utilities
using half2 = __half2;

// Warp-level 32 threads
#define WARP_SIZE 32

// Fast reciprocal square root
__device__ inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

// Warp-level reduction for sum
__device__ inline float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduction for sum of squares
__device__ inline float warpReduceSumSquares(float val) {
    val *= val;
    return warpReduceSum(val);
}

// Layer normalization kernel
__global__ void layerNormKernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int rows,
    int cols,
    float epsilon
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    if (row >= rows) return;

    // Shared memory for partial sums
    __shared__ float s_mean[WARP_SIZE];
    __shared__ float s_variance[WARP_SIZE];

    float sum = 0.0f;
    float sum_squares = 0.0f;

    // Compute sum and sum of squares in parallel
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = __half2float(input[row * cols + i]);
        sum += val;
        sum_squares += val * val;
    }

    // Warp-level reduction
    sum = warpReduceSum(sum);
    sum_squares = warpReduceSumSquares(sum_squares);

    // Store results in shared memory
    if (lane == 0) {
        s_mean[warpId] = sum / cols;
        s_variance[warpId] = sum_squares / cols - s_mean[warpId] * s_mean[warpId];
    }
    __syncthreads();

    // Compute final mean and variance
    float mean = 0.0f;
    float variance = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < blockDim.x / WARP_SIZE; ++i) {
            mean += s_mean[i];
            variance += s_variance[i];
        }
        variance = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    // Normalize and apply scale/shift
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = __half2float(input[row * cols + i]);
        val = (val - mean) * variance;
        val = val * gamma[i] + beta[i];
        output[row * cols + i] = __float2half(val);
    }
}

// Host function to launch kernel
extern "C" void layerNorm(
    const half* input,
    half* output,
    const float* gamma,
    const float* beta,
    int rows,
    int cols,
    float epsilon,
    cudaStream_t stream
) {
    assert(cols % WARP_SIZE == 0); // cols must be multiple of warp size
    const int threads = min(512, (cols + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    layerNormKernel<<<rows, threads, 0, stream>>>(input, output, gamma, beta, rows, cols, epsilon);
}