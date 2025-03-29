#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dotProduct(float *x, float *y, float *out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? x[idx] * y[idx] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1024;
    size_t size = n * sizeof(float);

    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);
    float *h_out = (float*)malloc(sizeof(float));

    for (int i = 0; i < n; i++) {
        h_x[i] = 2.0f;
        h_y[i] = 3.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dotProduct<<<blocks, threads, threads * sizeof(float)>>>(d_x, d_y, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Day 10 CUDA C: Dot product = %.1f (expected 6144.0)\n", *h_out);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);
    free(h_x); free(h_y); free(h_out);
    return 0;
}
