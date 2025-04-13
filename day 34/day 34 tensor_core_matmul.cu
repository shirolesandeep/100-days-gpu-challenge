#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>

// Use Tensor Cores for matrix multiply-accumulate operations
using namespace nvcuda::wmma;

// Matrix dimensions must be multiples of 16 for Tensor Cores
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// CUDA kernel that uses WMMA (Warp Matrix Multiply Accumulate)
__global__ void tensorCoreMatrixMul(half *A, half *B, float *C, int M, int N, int K) {
    // WMMA tile dimensions
    const int warpM = (WMMA_M * blockDim.y) / 32;
    const int warpN = (WMMA_N * blockDim.x) / 32;
    
    // Calculate warp and lane IDs
    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int lane = threadIdx.x % 32;
    
    // Calculate tile positions
    int warpRow = warpId / (blockDim.x / 32);
    int warpCol = warpId % (blockDim.x / 32);
    
    // Calculate starting row and column for this warp
    int row = blockIdx.y * warpM + warpRow * WMMA_M;
    int col = blockIdx.x * warpN + warpCol * WMMA_N;
    
    // Define WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator fragment
    fill_fragment(c_frag, 0.0f);
    
    // Loop over tiles
    for (int k = 0; k < K; k += WMMA_K) {
        // Check bounds
        if (row < M && k < K && col < N) {
            // Load fragments from global memory
            load_matrix_sync(a_frag, A + row * K + k, K);
            load_matrix_sync(b_frag, B + k * N + col, N);
            
            // Perform matrix multiplication
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Store result back to global memory
    if (row < M && col < N) {
        store_matrix_sync(C + row * N + col, c_frag, N, mem_row_major);
    }
}

// Host function for tensor core matrix multiplication
void tensorCoreMatMul(half* h_A, half* h_B, float* h_C, int M, int N, int K) {
    half *d_A, *d_B;
    float *d_C;
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, M * K * sizeof(half));
    cudaMalloc((void**)&d_B, K * N * sizeof(half));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    
    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // Set grid and block dimensions
    // Each warp computes a 16x16 output tile
    // Use 8x4 warps per block = 32 warps = 1024 threads
    dim3 threads(32, 32);
    dim3 grid((N + (WMMA_N * threads.x/32) - 1) / (WMMA_N * threads.x/32), 
              (M + (WMMA_M * threads.y/32) - 1) / (WMMA_M * threads.y/32));
    
    // Launch kernel
    tensorCoreMatrixMul<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

extern "C" {
    void cuda_tensor_matmul(float* h_A_float, float* h_B_float, float* h_C, int M, int N, int K) {
        // Convert float inputs to half precision
        half* h_A = (half*)malloc(M * K * sizeof(half));
        half* h_B = (half*)malloc(K * N * sizeof(half));
        
        // Manual conversion from float to half
        for (int i = 0; i < M * K; i++) {
            h_A[i] = __float2half(h_A_float[i]);
        }
        
        for (int i = 0; i < K * N; i++) {
            h_B[i] = __float2half(h_B_float[i]);
        }
        
        // Call the tensor core matrix multiplication
        tensorCoreMatMul(h_A, h_B, h_C, M, N, K);
        
        // Free temporary memory
        free(h_A);
        free(h_B);
    }
}
