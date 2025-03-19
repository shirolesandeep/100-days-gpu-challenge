#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixTranspose(float *A, float *B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        B[col * n + row] = A[row * n + col];
    }
}

int main() {
    int n = 32;
    size_t size = n * n * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = i + 1.0f;  // Fill with row number
        }
    }

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
    matrixTranspose<<<blocks, threads>>>(d_A, d_B, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("Day 9 CUDA C: B[0,1] = %.1f (expected 2.0)\n", h_B[1]);

    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
}
