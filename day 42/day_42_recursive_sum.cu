#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Base case threshold
#define THRESHOLD 1024

// Kernel for recursive array sum using dynamic parallelism
__global__ void recursiveSumKernel(int *input, int *output, int n, int level) {
    extern __shared__ int sdata[];

    // Thread and block indices
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Base case: if subproblem size is small, compute sum directly
    if (n <= THRESHOLD) {
        if (idx == 0) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                sum += input[i];
            }
            output[blockIdx.x] = sum;
        }
        return;
    }

    // Load data into shared memory
    if (idx < n) {
        sdata[tid] = input[idx];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // If this is the root level, store the result
    if (level == 0 && tid == 0) {
        output[blockIdx.x] = sdata[0];
        return;
    }

    // For non-base case, launch child kernels
    if (tid == 0 && sdata[0] != 0) {
        int *child_input = input;
        int *child_output;
        int child_n = n / gridDim.x;

        // Allocate memory for child output
        CUDA_CHECK(cudaMalloc(&child_output, gridDim.x * sizeof(int)));

        // Configure child kernel launch
        dim3 childBlock(256);
        dim3 childGrid((child_n + childBlock.x - 1) / childBlock.x);

        // Launch child kernel
        recursiveSumKernel<<<childGrid, childBlock, childBlock.x * sizeof(int)>>>(child_input, child_output, child_n, level + 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Sum the results from child kernels
        int sum = 0;
        for (int i = 0; i < gridDim.x; i++) {
            sum += child_output[i];
        }
        output[blockIdx.x] = sum;

        // Free child output memory
        CUDA_CHECK(cudaFree(child_output));
    }
}

int main() {
    // Array size (must be power of 2 for simplicity)
    const int n = 1 << 20; // 1M elements
    const int size = n * sizeof(int);

    // Host arrays
    int *h_input = (int*)malloc(size);
    int *h_output = (int*)malloc(sizeof(int));

    // Initialize input array
    for (int i = 0; i < n; i++) {
        h_input[i] = 1; // Example: all elements are 1
    }

    // Device arrays
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    recursiveSumKernel<<<grid, block, block.x * sizeof(int)>>>(d_input, d_output, n, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Print result
    printf("Sum of array: %d\n", h_output[0]);

    // Verify result
    int expected_sum = n; // Since each element is 1
    if (h_output[0] == expected_sum) {
        printf("Result is correct!\n");
    } else {
        printf("Result is incorrect! Expected %d\n", expected_sum);
    }

    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}