#include <stdio.h>
#include <cuda_runtime.h>

__global__ void elemMult(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) z[idx] = x[idx] * y[idx];
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);

    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    float *h_z = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_x[i] = 2.0f;
        h_y[i] = 3.0f;
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_z, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elemMult<<<blocks, threads>>>(d_x, d_y, d_z, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

    printf("Day 3 CUDA C: z[0] = %.1f (expected 6.0)\n", h_z[0]);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    free(h_x); free(h_y); free(h_z);
    return 0;
}
