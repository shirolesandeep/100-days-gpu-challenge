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

void cublasMatMul(int m, int n, int k, float* A, float* B, float* C) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform C = A * B
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));

    // Copy result back
    CHECK_CUDA(cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    int m = 4, n = 4, k = 4;
    float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float B[] = {1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1};
    float C[16] = {0};

    cublasMatMul(m, n, k, A, B, C);

    printf("Result matrix C:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C[i + j * m]);
        }
        printf("\n");
    }
    return 0;
}