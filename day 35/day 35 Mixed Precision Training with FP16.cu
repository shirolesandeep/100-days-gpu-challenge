// mixed_precision_training.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>
#include <mma.h>

using namespace nvcuda::wmma;

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << \
        cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// GEMM using Tensor Cores in mixed precision
__global__ void tensorCoreGemm(const half *A, const half *B, float *C, 
                              int M, int N, int K) {
    // Use fragments for Tensor Core operations
    fragment<matrix_a, 16, 16, 16, half, row_major> frag_A;
    fragment<matrix_b, 16, 16, 16, half, col_major> frag_B;
    fragment<accumulator, 16, 16, 16, float> frag_C;
    fragment<accumulator, 16, 16, 16, float> frag_D;
    
    // Calculate tile indices
    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warp_n = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (warp_m * 16 >= M || warp_n * 16 >= N) return;
    
    // Initialize accumulator with zeros
    fill_fragment(frag_C, 0.0f);
    
    // Calculate offsets
    int a_offset = warp_m * 16 * K;
    int b_offset = warp_n * 16;
    
    // Loop over the K dimension tile by tile
    for (int k = 0; k < K; k += 16) {
        // Boundary check for K
        if (k + 16 > K) break;
        
        // Load matrix fragments from memory
        load_matrix_sync(frag_A, A + a_offset + k, K);
        load_matrix_sync(frag_B, B + b_offset + k * N, N);
        
        // Perform the matrix multiplication
        mma_sync(frag_C, frag_A, frag_B, frag_C);
    }
    
    // Store the result back to C
    store_matrix_sync(C + warp_m * 16 * N + warp_n * 16, frag_C, N, mem_row_major);
}

// Wrapper for mixed precision training
class MixedPrecisionTrainer {
public:
    MixedPrecisionTrainer(int m, int n, int k) : M(m), N(n), K(k) {
        // Initialize CUBLAS
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        
        // Allocate memory
        size_t a_size = M * K * sizeof(half);
        size_t b_size = K * N * sizeof(half);
        size_t c_size = M * N * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_A, a_size));
        CUDA_CHECK(cudaMalloc(&d_B, b_size));
        CUDA_CHECK(cudaMalloc(&d_C, c_size));
        CUDA_CHECK(cudaMalloc(&d_gradOutput, c_size));
        CUDA_CHECK(cudaMalloc(&d_gradInput, a_size));
        
        // Create scaling factors for mixed precision
        alpha = 1.0f;
        beta = 0.0f;
    }
    
    ~MixedPrecisionTrainer() {
        // Clean up
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_gradOutput);
        cudaFree(d_gradInput);
    }
    
    // Forward pass using cuBLAS with Tensor Cores
    void forward() {
        // Use cuBLAS GEMM with tensor cores
        cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, CUDA_R_16F, N,
                    d_A, CUDA_R_16F, K,
                    &beta,
                    d_C, CUDA_R_32F, N,
                    CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    // Custom forward using our kernel
    void forwardCustom() {
        dim3 gridDim((M + 15) / 16, (N + 15) / 16);
        dim3 blockDim(32, 2); // 2 warps per block
        
        tensorCoreGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    
    // Backward pass for gradients
    void backward() {
        // Compute gradients with respect to inputs
        // For simplicity, only compute dA = gradOutput * B^T
        cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    K, M, N,
                    &alpha,
                    d_B, CUDA_R_16F, N,
                    d_gradOutput, CUDA_R_32F, N,
                    &beta,
                    d_gradInput, CUDA_R_16F, K,
                    CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    // Set data for computation
    void setInputs(half* h_A, half* h_B) {
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    }
    
    // Set gradients for backward pass
    void setGradOutput(float* h_gradOutput) {
        CUDA_CHECK(cudaMemcpy(d_gradOutput, h_gradOutput, M * N * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Get results
    void getOutput(float* h_C) {
        CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Get gradients
    void getGradInput(half* h_gradInput) {
        CUDA_CHECK(cudaMemcpy(h_gradInput, d_gradInput, M * K * sizeof(half), cudaMemcpyDeviceToHost));
    }
    
private:
    int M, N, K;
    cublasHandle_t handle;
    half *d_A, *d_B;
    float *d_C, *d_gradOutput;
    half *d_gradInput;
    float alpha, beta;
};

int main() {
    // Example dimensions
    int M = 4096;
    int N = 4096;
    int K = 4096;
    
    // Initialize trainer
    MixedPrecisionTrainer trainer(M, N, K);
    
    // Allocate and initialize input data
    half *h_A = new half[M * K];
    half *h_B = new half[K * N];
    float *h_C = new float[M * N];
    float *h_gradOutput = new float[M * N];
    half *h_gradInput = new half[M * K];
    
    // Initialize data with some values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(0.1f * (i % 10));
    }
    
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(0.01f * (i % 10));
    }
    
    for (int i = 0; i < M * N; i++) {
        h_gradOutput[i] = 0.001f * (i % 10);
    }
    
    // Set inputs
    trainer.setInputs(h_A, h_B);
    trainer.setGradOutput(h_gradOutput);
    
    // Run forward and backward passes
    trainer.forward();
    trainer.backward();
    
    // Get results
    trainer.getOutput(h_C);
    trainer.getGradInput(h_gradInput);
    
    // Verify some results (just examples)
    std::cout << "First few elements of output C:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_gradOutput;
    delete[] h_gradInput;
    
    return 0;
}