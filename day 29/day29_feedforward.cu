#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ float gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x3)));
}

__global__ void feedforward_kernel(float* input, float* W1, float* W2, float* output, 
                                  int batch_size, int seq_len, int d_model, int d_ff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * seq_len * d_model) {
        int row = idx / d_model;
        float hidden = 0.0f;
        for (int i = 0; i < d_ff; i++) {
            hidden += input[row * d_model + i] * W1[i * d_model + idx % d_model];
        }
        hidden = gelu(hidden);
        output[idx] = hidden * W2[(idx % d_model) * d_ff + idx % d_ff];
    }
}

void launch_feedforward(float* input, float* W1, float* W2, float* output, 
                       int batch_size, int seq_len, int d_model, int d_ff) {
    int threads = 256;
    int blocks = (batch_size * seq_len * d_model + threads - 1) / threads;
    feedforward_kernel<<<blocks, threads>>>(input, W1, W2, output, batch_size, seq_len, d_model, d_ff);
}



