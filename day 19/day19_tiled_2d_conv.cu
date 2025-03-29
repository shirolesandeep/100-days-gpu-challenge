#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define TILE_SIZE 16

__global__ void tiled2DConvKernel(
    const float* input, const float* kernel, float* output,
    int width, int height, int kernel_size) {
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];  // +2 for kernel overlap (assuming kernel_size <= 3)

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    int k_half = kernel_size / 2;

    // Load tile with boundary checks
    int input_row = row - k_half;
    int input_col = col - k_half;
    if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
        tile[ty][tx] = input[input_row * width + input_col];
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    // Compute convolution
    if (row < height && col < width && tx < blockDim.x && ty < blockDim.y) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum += tile[ty + i][tx + j] * kernel[(kernel_size - 1 - i) * kernel_size + (kernel_size - 1 - j)];
            }
        }
        output[row * width + col] = sum;
    }
}

void tiled2DConv(
    int width, int height, int kernel_size, float* h_input, float* h_kernel, float* h_output) {
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, width * height * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    tiled2DConvKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, kernel_size);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
}

int main() {
    int width = 4, height = 4, kernel_size = 3;
    float h_input[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float h_kernel[] = {1, 0, -1, 0, 0, 0, -1, 0, 1};  // Simple edge detection kernel
    float h_output[16] = {0};

    tiled2DConv(width, height, kernel_size, h_input, h_kernel, h_output);

    printf("2D Convolution output:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", h_output[i * width + j]);
        }
        printf("\n");
    }
    return 0;
}
