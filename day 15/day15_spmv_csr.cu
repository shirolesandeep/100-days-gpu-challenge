#include <stdio.h>
#include <cuda_runtime.h>

#define N 512  // Matrix size (N x N)
#define DENSITY 0.1  // Sparsity: 10% non-zero elements

// CUDA kernel for SpMV using CSR format
__global__ void spmvCSR(float *values, int *col_indices, int *row_ptr, float *x, float *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        for (int i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_indices[i]];
        }
        y[row] = sum;
    }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = N;
    int nnz = (int)(N * N * DENSITY);  // Number of non-zero elements
    
    // Host data: CSR format (values, col_indices, row_ptr) and vectors x, y
    float *h_values = (float*)malloc(nnz * sizeof(float));
    int *h_col_indices = (int*)malloc(nnz * sizeof(int));
    int *h_row_ptr = (int*)malloc((n + 1) * sizeof(int));
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));
    
    // Initialize sparse matrix in CSR format (simple pattern for demo)
    int idx = 0;
    h_row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        int non_zeros_in_row = 0;
        for (int j = 0; j < n; j++) {
            if ((i + j) % 10 == 0 && idx < nnz) {  // Sparse pattern
                h_values[idx] = (float)(i + j + 1);
                h_col_indices[idx] = j;
                idx++;
                non_zeros_in_row++;
            }
        }
        h_row_ptr[i + 1] = h_row_ptr[i] + non_zeros_in_row;
    }
    nnz = idx;  // Update nnz based on actual non-zeros
    
    // Initialize vector x
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)(i % 10);
    }
    
    // Device data
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptr;
    checkCudaError(cudaMalloc(&d_values, nnz * sizeof(float)), "CUDA malloc d_values failed");
    checkCudaError(cudaMalloc(&d_col_indices, nnz * sizeof(int)), "CUDA malloc d_col_indices failed");
    checkCudaError(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)), "CUDA malloc d_row_ptr failed");
    checkCudaError(cudaMalloc(&d_x, n * sizeof(float)), "CUDA malloc d_x failed");
    checkCudaError(cudaMalloc(&d_y, n * sizeof(float)), "CUDA malloc d_y failed");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice), "CUDA memcpy to d_values failed");
    checkCudaError(cudaMemcpy(d_col_indices, h_col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice), "CUDA memcpy to d_col_indices failed");
    checkCudaError(cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice), "CUDA memcpy to d_row_ptr failed");
    checkCudaError(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice), "CUDA memcpy to d_x failed");
    
    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    spmvCSR<<<blocksPerGrid, threadsPerBlock>>>(d_values, d_col_indices, d_row_ptr, d_x, d_y, n);
    cudaEventRecord(stop);
    
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost), "CUDA memcpy to host failed");
    
    // Print execution time
    float milliseconds = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %.4f ms\n", milliseconds);
    
    // Verify results (print first few elements)
    printf("First 5 elements of result vector y:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", h_y[i]);
    }
    printf("\n");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_values);
    cudaFree(d_col_indices);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_values);
    free(h_col_indices);
    free(h_row_ptr);
    free(h_x);
    free(h_y);
    
    return 0;
}
