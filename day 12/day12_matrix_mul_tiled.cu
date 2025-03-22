#include <stdio.h>
#include <cuda_runtime.h>

#define N 512  // Matrix size (N x N)
#define TILE_SIZE 32  // Tile size for shared memory

// CUDA kernel for tiled matrix multiplication using shared memory
__global__ void matrixMulTiled(float *A, float *B, float *C, int n) {
    // Shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        tileA[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < n && col < n) {
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "CUDA memcpy to host failed");
    
    // Print execution time
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %.4f ms\n", milliseconds);
    
    // Verify results (print a few elements)
    printf("Sample results (top-left 3x3):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%.1f ", h_C[i * n + j]);
        }
        printf("\n");
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
