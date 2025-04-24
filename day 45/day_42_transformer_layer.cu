#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Constants
const int BATCH_SIZE = 32;
const int SEQ_LEN = 128;
const int EMBED_DIM = 512;
const int NUM_HEADS = 8;
const int HEAD_DIM = EMBED_DIM / NUM_HEADS;
const int FF_DIM = 2048;

// Convert float to half
__host__ __device__ half float_to_half(float f) {
    return __float2half(f);
}

// Scaled Dot-Product Attention kernel with shared memory
__global__ void scaled_dot_product_attention(
    half* q, half* k, half* v, half* output,
    int batch_size, int seq_len, int head_dim, int num_heads
) {
    extern __shared__ half shared_mem[];
    int tid = threadIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    
    // Shared memory partitions
    half* s_q = shared_mem;
    half* s_k = s_q + seq_len * head_dim;
    half* s_v = s_k + seq_len * head_dim;
    half* s_scores = s_v + seq_len * head_dim;
    
    // Global memory indices
    int q_offset = batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
    int k_offset = q_offset;
    int v_offset = q_offset;
    
    // Load Q, K, V to shared memory (coalesced access)
    for (int i = tid; i < seq_len * head_dim; i += blockDim.x) {
        s_q[i] = q[q_offset + i];
        s_k[i] = k[k_offset + i];
        s_v[i] = v[v_offset + i];
    }
    __syncthreads();
    
    // Compute attention scores
    if (tid < seq_len) {
        for (int j = 0; j < seq_len; j++) {
            half score = 0;
            for (int d = 0; d < head_dim; d++) {
                score += __hmul(s_q[tid * head_dim + d], s_k[j * head_dim + d]);
            }
            score = __hdiv(score, __float2half(sqrtf((float)head_dim)));
            s_scores[tid * seq_len + j] = score;
        }
    }
    __syncthreads();
    
    // Softmax (simplified for brevity)
    if (tid < seq_len) {
        half sum = 0;
        for (int j = 0; j < seq_len; j++) {
            s_scores[tid * seq_len + j] = hexp(s_scores[tid * seq_len + j]);
            sum += s_scores[tid * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++) {
            s_scores[tid * seq_len + j] = __hdiv(s_scores[tid * seq_len + j], sum);
        }
    }
    __syncthreads();
    
    // Compute output
    if (tid < seq_len) {
        for (int d = 0; d < head_dim; d++) {
            half out = 0;
            for (int j = 0; j < seq_len; j++) {
                out += __hmul(s_scores[tid * seq_len + j], s_v[j * head_dim + d]);
            }
            output[q_offset + tid * head_dim + d] = out;
        }
    }
}

// Feed-Forward Network kernel
__global__ void feed_forward(
    half* input, half* output, half* w1, half* w2,
    half* b1, half* b2, int batch_size, int seq_len, int embed_dim, int ff_dim
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = tid;
    
    if (idx < batch_size * seq_len) {
        half hidden[FF_DIM];
        // First layer
        for (int i = 0; i < ff_dim; i++) {
            hidden[i] = b1[i];
            for (int j = 0; j < embed_dim; j++) {
                hidden[i] += __hmul(input[idx * embed_dim + j], w1[j * ff_dim + i]);
            }
            hidden[i] = hidden[i] > 0 ? hidden[i] : 0; // ReLU
        }
        
        // Second layer
        for (int i = 0; i < embed_dim; i++) {
            half out = b2[i];
            for (int j = 0; j < ff_dim; j++) {
                out += __hmul(hidden[j], w2[j * embed_dim + i]);
            }
            output[idx * embed_dim + i] = out;
        }
    }
}

// Main transformer layer function
void transformer_layer(
    half* input, half* output, half* qkv_weights, half* attn_output_weights,
    half* ff_w1, half* ff_w2, half* ff_b1, half* ff_b2,
    cublasHandle_t cublas_handle, cudaStream_t stream
) {
    cudaEvent_t start, stop;
    float elapsed_time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Profile memory bandwidth
    CUDA_CHECK(cudaEventRecord(start, stream));
    
    // QKV projection (using cuBLAS for matrix multiplication)
    half* qkv_output;
    CUDA_CHECK(cudaMalloc(&qkv_output, BATCH_SIZE * SEQ_LEN * EMBED_DIM * 3 * sizeof(half)));
    
    const half alpha = float_to_half(1.0f);
    const half beta = float_to_half(0.0f);
    cublasSetStream(cublas_handle, stream);
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                EMBED_DIM * 3, BATCH_SIZE * SEQ_LEN, EMBED_DIM,
                &alpha, qkv_weights, EMBED_DIM * 3,
                input, EMBED_DIM, &beta,
                qkv_output, EMBED_DIM * 3);
    
    // Split QKV
    half *q = qkv_output;
    half *k = qkv_output + BATCH_SIZE * SEQ_LEN * EMBED_DIM;
    half *v = qkv_output + 2 * BATCH_SIZE * SEQ_LEN * EMBED_DIM;
    
    // Multi-head attention
    half* attn_output;
    CUDA_CHECK(cudaMalloc(&attn_output, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half)));
    
    dim3 attn_grid(1, NUM_HEADS, BATCH_SIZE);
    dim3 attn_block(SEQ_LEN);
    size_t shared_mem_size = (3 * SEQ_LEN * HEAD_DIM + SEQ_LEN * SEQ_LEN) * sizeof(half);
    scaled_dot_product_attention<<<attn_grid, attn_block, shared_mem_size, stream>>>(
        q, k, v, attn_output, BATCH_SIZE, SEQ_LEN, HEAD_DIM, NUM_HEADS
    );
    
    // Attention output projection
    half* proj_output;
    CUDA_CHECK(cudaMalloc(&proj_output, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half)));
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                EMBED_DIM, BATCH_SIZE * SEQ_LEN, EMBED_DIM,
                &alpha, attn_output_weights, EMBED_DIM,
                attn_output, EMBED_DIM, &beta,
                proj_output, EMBED_DIM);
    
    // Residual connection
    half* residual;
    CUDA_CHECK(cudaMalloc(&residual, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half)));
    cudaMemcpyAsync(residual, input, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half),
                   cudaMemcpyDeviceToDevice, stream);
    
    // Feed-forward
    half* ff_output;
    CUDA_CHECK(cudaMalloc(&ff_output, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half)));
    
    dim3 ff_grid((BATCH_SIZE * SEQ_LEN + 255) / 256);
    dim3 ff_block(256);
    feed_forward<<<ff_grid, ff_block, 0, stream>>>(
        proj_output, ff_output, ff_w1, ff_w2, ff_b1, ff_b2,
        BATCH_SIZE, SEQ_LEN, EMBED_DIM, FF_DIM
    );
    
    // Final residual connection
    cudaMemcpyAsync(output, ff_output, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half),
                   cudaMemcpyDeviceToDevice, stream);
    
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    
    // Calculate memory bandwidth
    size_t bytes_read = (BATCH_SIZE * SEQ_LEN * EMBED_DIM * 3 + // QKV input
                        BATCH_SIZE * SEQ_LEN * EMBED_DIM * 3 + // QKV weights
                        BATCH_SIZE * SEQ_LEN * EMBED_DIM +    // Attention output
                        BATCH_SIZE * SEQ_LEN * EMBED_DIM +    // FF input
                        EMBED_DIM * FF_DIM +                  // FF weights 1
                        FF_DIM * EMBED_DIM) * sizeof(half);   // FF weights 2
    size_t bytes_written = (BATCH_SIZE * SEQ_LEN * EMBED_DIM * 4) * sizeof(half); // Various outputs
    float bandwidth = (bytes_read + bytes_written) / (elapsed_time / 1000.0f) / 1e9;
    
    printf("Transformer Layer Bandwidth: %.2f GB/s\n", bandwidth);
    
    // Cleanup
    CUDA_CHECK(cudaFree(qkv_output));
    CUDA_CHECK(cudaFree(attn_output));
    CUDA_CHECK(cudaFree(proj_output));
    CUDA_CHECK(cudaFree(residual));
    CUDA_CHECK(cudaFree(ff_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    // Allocate memory
    half *d_input, *d_output, *d_qkv_weights, *d_attn_output_weights;
    half *d_ff_w1, *d_ff_w2, *d_ff_b1, *d_ff_b2;
    
    CUDA_CHECK(cudaMalloc(&d_input, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_qkv_weights, EMBED_DIM * EMBED_DIM * 3 * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_attn_output_weights, EMBED_DIM * EMBED_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ff_w1, EMBED_DIM * FF_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ff_w2, FF_DIM * EMBED_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ff_b1, FF_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ff_b2, EMBED_DIM * sizeof(half)));
    
    // Initialize weights (simplified)
    half* h_input = (half*)malloc(BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half));
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * EMBED_DIM; i++) {
        h_input[i] = float_to_half((float)rand() / RAND_MAX);
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, BATCH_SIZE * SEQ_LEN * EMBED_DIM * sizeof(half),
                         cudaMemcpyHostToDevice));
    
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Run transformer layer
    transformer_layer(
        d_input, d_output, d_qkv_weights, d_attn_output_weights,
        d_ff_w1, d_ff_w2, d_ff_b1, d_ff_b2,
        cublas_handle, stream
    );
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_qkv_weights));
    CUDA_CHECK(cudaFree(d_attn_output_weights));
    CUDA_CHECK(cudaFree(d_ff_w1));
    CUDA_CHECK(cudaFree(d_ff_w2));
    CUDA_CHECK(cudaFree(d_ff_b1));
    CUDA_CHECK(cudaFree(d_ff_b2));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cublasDestroy(cublas_handle);
    free(h_input);
    
    return 0;
}