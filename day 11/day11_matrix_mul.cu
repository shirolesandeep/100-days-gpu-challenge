%%writefile day11_matrix_mul.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define N 512  // Matrix size (N x N)
#define TILE_SIZE 16  // Tile size (not used in basic version but defined for future optimization)

// CUDA kernel for basic matrix multiplication
__global__ void matrixMul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = N;
    size_t bytes = n * n * sizeof(float);
    
    // Host matrices
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)(i % 10);  // Simple pattern
        h_B[i] = (float)((i + 1) % 10);
    }
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, bytes), "CUDA malloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, bytes), "CUDA malloc d_B failed");
    checkCudaError(cudaMalloc(&d_C, bytes), "CUDA malloc d_C failed");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "CUDA memcpy to d_A failed");
    checkCudaError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "CUDA memcpy to d_B failed");
    
    // Configure kernel launch
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "CUDA memcpy to host failed");
    
    // Verify results (print a few elements)
    printf("Sample results (top-left 3x3):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%.1f ", h_C[i * n + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}