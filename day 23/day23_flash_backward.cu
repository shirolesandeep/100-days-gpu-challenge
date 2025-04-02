#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(err); \
    }

__global__ void flashAttentionBackward(float* dO, float* Q, float* K, float* V, 
                                      float* P, float* dQ, float* dK, float* dV, 
                                      int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared_mem[];
    float* dS_tile = shared_mem;

    if (row >= N || col >= N) return;

    // Compute dS = dO * V^T
    float dS = 0.0f;
    for (int k = 0; k < d; k++) {
        dS += dO[row * d + k] * V[col * d + k];
    }
    dS_tile[threadIdx.y * blockDim.x + threadIdx.x] = dS * P[row * N + col];
    __syncthreads();

    // Gradient w.r.t. Q: dQ = dS * K
    if (col < d) {
        float grad_q = 0.0f;
        for (int k = 0; k < N; k++) {
            grad_q += dS_tile[threadIdx.y * blockDim.x + k] * K[k * d + col];
        }
        dQ[row * d + col] = grad_q * scale;
    }

    // Gradient w.r.t. K: dK = dS^T * Q
    if (col < d) {
        float grad_k = 0.0f;
        for (int k = 0; k < N; k++) {
            grad_k += dS_tile[k * blockDim.x + threadIdx.y] * Q[k * d + col];
        }
        dK[row * d + col] = grad_k * scale;
    }

    // Gradient w.r.t. V: dV = P^T * dO
    if (col < d) {
        float grad_v = 0.0f;
        for (int k = 0; k < N; k++) {
            grad_v += P[k * N + row] * dO[k * d + col];
        }
        dV[row * d + col] = grad_v;
    }
}

void launchFlashAttentionBackward(float* dO, float* Q, float* K, float* V, 
                                 float* P, float* dQ, float* dK, float* dV, 
                                 int N, int d) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    size_t shared_mem_size = block.x * block.y * sizeof(float);
    flashAttentionBackward<<<grid, block, shared_mem_size>>>(dO, Q, K, V, P, dQ, dK, dV, N, d, 1.0f / sqrtf(d));
    CHECK_CUDA_ERROR(cudaGetLastError());
}
