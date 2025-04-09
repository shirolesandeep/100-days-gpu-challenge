#include <stdio.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    // Use XOR mode of shuffle to perform butterfly reduction
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// Kernel using warp-level reduction
__global__ void reduceKernel(float *input, float *output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;
    
    // Each thread loads one element
    float sum = 0.0f;
    if (tid < n) {
        sum = input[tid];
    }
    
    // Perform warp-level reduction
    sum = warpReduceSum(sum);
    
    // Only the first thread in each warp writes the result
    if (laneId == 0) {
        atomicAdd(&output[blockIdx.x], sum);
    }
}

// Host function to perform the reduction
float reduce(float *h_input, int n) {
    float *d_input, *d_output, h_output = 0.0f;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, n * sizeof(float));
    
    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc((void**)&d_output, blocksPerGrid * sizeof(float));
    cudaMemset(d_output, 0, blocksPerGrid * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    
    // Copy partial results back to host
    float *h_partial = (float*)malloc(blocksPerGrid * sizeof(float));
    cudaMemcpy(h_partial, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Sum partial results on host
    for (int i = 0; i < blocksPerGrid; i++) {
        h_output += h_partial[i];
    }
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_partial);
    
    return h_output;
}

extern "C" {
    float cuda_reduce(float *h_input, int n) {
        return reduce(h_input, n);
    }
}
