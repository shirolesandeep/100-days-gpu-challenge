#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorScale(float *x, float *y, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a * x[idx];
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);
    float a = 5.0f;

    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_x[i] = 2.0f;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vectorScale<<<blocks, threads>>>(d_x, d_y, a, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    printf("Day 4 CUDA C: y[0] = %.1f (expected 10.0)\n", h_y[0]);

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y);
    return 0;
}
