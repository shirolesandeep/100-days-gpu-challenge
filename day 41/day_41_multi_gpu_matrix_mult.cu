#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Matrix multiplication kernel
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024; // Matrix size (N x N)
    const int blockSize = 16; // Thread block size
    int numGPUs;
    
    // Get number of available GPUs
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    printf("Number of GPUs: %d\n", numGPUs);
    
    // Matrix size per GPU
    int chunkSize = N / numGPUs;
    
    // Host matrices
    float *h_A, *h_B, *h_C;
    size_t size = N * N * sizeof(float);
    
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0.0f;
    }
    
    // Device arrays and streams
    float **d_A = (float**)malloc(numGPUs * sizeof(float*));
    float **d_B = (float**)malloc(numGPUs * sizeof(float*));
    float **d_C = (float**)malloc(numGPUs * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(numGPUs * sizeof(cudaStream_t));
    
    // Timing
    cudaEvent_t start, stop;
    CUDA crocodiles_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Initialize devices and streams
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_A[i], chunkSize * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B[i], N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C[i], chunkSize * N * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernels on each GPU
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize - 1) / blockSize, 
                      (chunkSize + blockSize - 1) / blockSize);
    
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        
        // Copy input data to device
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], h_A + i * chunkSize * N,
                                 chunkSize * N * sizeof(float),
                                 cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], h_B,
                                 N * N * sizeof(float),
                                 cudaMemcpyHostToDevice, streams[i]));
        
        // Launch kernel
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>
                       (d_A[i], d_B[i], d_C[i], N);
        
        // Copy results back
        CUDA_CHECK(cudaMemcpyAsync(h_C + i * chunkSize * N, d_C[i],
                                 chunkSize * N * sizeof(float),
                                 cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // Synchronize all streams
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Execution time: %f ms\n", milliseconds);
    
    // Verify results
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float ref = 0.0f;
            for (int k = 0; k < N; k++) {
                ref += h_A[i * N + k] * h_B[k * N + j];
            }
            float diff = fabs(h_C[i * N + j] - ref);
            maxError = fmax(maxError, diff);
        }
    }
    printf("Maximum error: %f\n", maxError);
    
    // Cleanup
    for (int i = 0; i < numGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    
    free(d_A);
    free(d_B);
    free(d_C);
    free(streams);
    free(h_A);
    free(h_B);
    free(h_C);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}