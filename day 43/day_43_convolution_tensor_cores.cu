#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// WMMA fragment sizes for Tensor Cores (16x16x16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Tile sizes for the convolution
#define TILE_WIDTH 32
#define KERNEL_SIZE 3
#define HALF_KERNEL (KERNEL_SIZE / 2)

// CUDA kernel for 2D convolution using Tensor Cores
__global__ void conv2dTensorCoreKernel(
    const half* __restrict__ input,
    const half* __restrict__ kernel,
    half* __restrict__ output,
    int width, int height, int channels,
    int out_width, int out_height
) {
    // Shared memory for input and kernel tiles
    __shared__ half s_input[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];
    __shared__ half s_kernel[KERNEL_SIZE][KERNEL_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // Output coordinates
    int row = by * (TILE_WIDTH - KERNEL_SIZE + 1) + ty;
    int col = bx * (TILE_WIDTH - KERNEL_SIZE + 1) + tx;

    // WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize accumulator fragment to zero
    nvcuda::wmma::fill_fragment(c_frag, __float2half(0.0f));

    // Load kernel into shared memory (single channel for simplicity)
    if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
        s_kernel[ty][tx] = kernel[ty * KERNEL_SIZE + tx];
    }
    __syncthreads();

    // Load input tile into shared memory
    int input_row = by * (TILE_WIDTH - KERNEL_SIZE + 1) - HALF_KERNEL + ty;
    int input_col = bx * (TILE_WIDTH - KERNEL_SIZE + 1) - HALF_KERNEL + tx;

    if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
        s_input[ty][tx] = input[input_row * width + input_col];
    } else {
        s_input[ty][tx] = __float2half(0.0f);
    }
    __syncthreads();

    // Perform convolution using Tensor Cores
    if (row < out_height && col < out_width && ty < WMMA_M && tx < WMMA_N) {
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                // Load input and kernel fragments
                nvcuda::wmma::load_matrix_sync(a_frag, s_input[ty + ky] + tx + kx, TILE_WIDTH + KERNEL_SIZE - 1);
                nvcuda::wmma::fill_fragment(b_frag, s_kernel[ky][kx]);
                // Perform matrix multiply-accumulate
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        // Store result to global memory
        nvcuda::wmma::store_matrix_sync(output + row * out_width + col, c_frag, out_width, nvcuda::wmma::mem_row_major);
    }
}

// Host function to launch the kernel
void launchConv2dTensorCore(
    const half* input,
    const half* kernel,
    half* output,
    int width, int height, int channels,
    int kernel_size
) {
    int out_width = width - kernel_size + 1;
    int out_height = height - kernel_size + 1;

    // Grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(
        (out_width + (TILE_WIDTH - kernel_size + 1) - 1) / (TILE_WIDTH - kernel_size + 1),
        (out_height + (TILE_WIDTH - kernel_size + 1) - 1) / (TILE_WIDTH - kernel_size + 1)
    );

    // Launch kernel
    conv2dTensorCoreKernel<<<gridDim, blockDim>>>(
        input, kernel, output,
        width, height, channels,
        out_width, out_height
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Main function for testing
int main() {
    const int width = 64;
    const int height = 64;
    const int channels = 1;
    const int kernel_size = KERNEL_SIZE;

    // Allocate host memory
    half *h_input = new half[width * height * channels];
    half *h_kernel = new half[kernel_size * kernel_size];
    half *h_output = new half[(width - kernel_size + 1) * (height - kernel_size + 1)];

    // Initialize input and kernel with sample data
    for (int i = 0; i < width * height * channels; i++) {
        h_input[i] = __float2half(1.0f);
    }
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        h_kernel[i] = __float2half(0.1111f); // Simple averaging kernel
    }

    // Allocate device memory
    half *d_input, *d_kernel, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, width * height * channels * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, (width - kernel_size + 1) * (height - kernel_size + 1) * sizeof(half)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, width * height * channels * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(half), cudaMemcpyHostToDevice));

    // Launch convolution
    launchConv2dTensorCore(d_input, d_kernel, d_output, width, height, channels, kernel_size);

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, (width - kernel_size + 1) * (height - kernel_size + 1) * sizeof(half), cudaMemcpyDeviceToHost));

    // Print a sample output value
    printf("Sample output[0]: %f\n", __half2float(h_output[0]));

    // Clean up
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    return 0;
}