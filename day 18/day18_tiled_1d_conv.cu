#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define TILE_SIZE 32

__global__ void tiled1DConvKernel(
    const float* input, const float* kernel, float* output,
    int input_len, int kernel_len) {
    __shared__ float tile[TILE_SIZE + 4];  // +4 for kernel overlap (assuming kernel_len <= 5)

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int output_idx = gid - (kernel_len / 2);  // Center the kernel

    // Load tile with boundary checks
    if (gid < input_len + kernel_len - 1) {
        tile[tid] = (gid < input_len) ? input[gid] : 0.0f;
    } else {
        tile[tid] = 0.0f;
    }
    __syncthreads();

    // Compute convolution
    if (output_idx >= 0 && output_idx < input_len && tid < blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_len; i++) {
            int tile_idx = tid + i;
            if (tile_idx < TILE_SIZE + kernel_len - 1) {
                sum += tile[tile_idx] * kernel[kernel_len - 1 - i];
            }
        }
        output[output_idx] = sum;
    }
}

void tiled1DConv(
    int input_len, int kernel_len, float* h_input, float* h_kernel, float* h_output) {
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, input_len * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_len * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = TILE_SIZE;
    int gridSize = (input_len + kernel_len - 1 + blockSize - 1) / blockSize;
    tiled1DConvKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, input_len, kernel_len);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, input_len * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
}

int main() {
    int input_len = 8;
    int kernel_len = 3;
    float h_input[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_kernel[] = {1, 0, -1};  // Simple edge detection kernel
    float h_output[8] = {0};

    tiled1DConv(input_len, kernel_len, h_input, h_kernel, h_output);

    printf("Convolution output:\n");
    for (int i = 0; i < input_len; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    return 0;
}