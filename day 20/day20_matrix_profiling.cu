#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
}

void profileMatrixMul(int m, int n, int k, float* A, float* B, float* C) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warm-up run
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));

    // Profile
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 10; i++) {  // Average over 10 runs
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("cuBLAS Matrix Mul (%dx%d x %dx%d): %f ms (avg over 10 runs)\n", m, k, k, n, milliseconds / 10);

    CHECK_CUDA(cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    int m = 1024, n = 1024, k = 1024;
    float* A = (float*)malloc(m * k * sizeof(float));
    float* B = (float*)malloc(k * n * sizeof(float));
    float* C = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices (simple example)
    for (int i = 0; i < m * k; i++) A[i] = (float)(i % 10);
    for (int i = 0; i < k * n; i++) B[i] = (float)(i % 5);

    profileMatrixMul(m, n, k, A, B, C);

    free(A); free(B); free(C);
    return 0;
}
