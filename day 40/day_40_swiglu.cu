#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Kernel constants
#define WARP_SIZE 32
#define BLOCK_SIZE 256

/**
 * SwiGLU activation function: Swish(x*W1) ⊙ (x*W2)
 * where Swish(x) = x * sigmoid(x)
 */

// Sigmoid function
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Swish function: x * sigmoid(x)
__device__ __forceinline__ float swish(float x) {
    return x * sigmoid(x);
}

/**
 * Warp-level optimized SwiGLU activation function
 * This kernel uses warp-level primitives for better performance
 * Each warp processes a set of contiguous elements
 */
__global__ void swiGLU_warp_optimized(
    const float* input,         // Input tensor [batch_size, seq_len, in_features]
    const float* weights_gate,  // Gate weights [in_features, out_features]
    const float* weights_up,    // Up-projection weights [in_features, out_features]
    float* output,              // Output tensor [batch_size, seq_len, out_features]
    int batch_size,
    int seq_len,
    int in_features,
    int out_features
) {
    // Calculate total elements in the output
    int total_elements = batch_size * seq_len * out_features;
    
    // Get warp and lane IDs
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Calculate thread's global position
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_tid = global_tid / WARP_SIZE;
    
    // Each warp processes WARP_SIZE elements at a time
    for (int idx = warp_tid * WARP_SIZE + lane_id; idx < total_elements; idx += (gridDim.x * blockDim.x / WARP_SIZE) * WARP_SIZE) {
        // Convert flat index to 3D position
        int b = idx / (seq_len * out_features);
        int s = (idx % (seq_len * out_features)) / out_features;
        int o = idx % out_features;
        
        if (b < batch_size && s < seq_len && o < out_features) {
            // Calculate input base index
            int input_base = (b * seq_len + s) * in_features;
            
            // Calculate the gate activation (Swish) and the linear projection
            float gate_val = 0.0f;
            float up_val = 0.0f;
            
            // Each thread computes a dot product for its assigned output element
            for (int i = 0; i < in_features; i++) {
                float input_val = input[input_base + i];
                gate_val += input_val * weights_gate[i * out_features + o];
                up_val += input_val * weights_up[i * out_features + o];
            }
            
            // Apply Swish to gate value: gate_val * sigmoid(gate_val)
            float swish_val = swish(gate_val);
            
            // Final SwiGLU: Swish(gate) * up
            output[idx] = swish_val * up_val;
        }
    }
}

/**
 * Version with warp shuffle to accelerate dot products
 * Uses warp shuffle to share data between threads in a warp
 */
__global__ void swiGLU_warp_shuffle(
    const float* input,
    const float* weights_gate,
    const float* weights_up,
    float* output,
    int batch_size, 
    int seq_len,
    int in_features,
    int out_features
) {
    // Get warp and lane IDs
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Calculate global warp index
    int global_warp_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    
    // Each warp processes one output element
    int elements_per_grid = gridDim.x * (blockDim.x / WARP_SIZE);
    
    for (int idx = global_warp_id; idx < batch_size * seq_len * out_features; idx += elements_per_grid) {
        // Convert flat index to 3D position
        int b = idx / (seq_len * out_features);
        int s = (idx % (seq_len * out_features)) / out_features;
        int o = idx % out_features;
        
        if (b < batch_size && s < seq_len && o < out_features) {
            // Calculate input base index
            int input_base = (b * seq_len + s) * in_features;
            
            // Each thread in the warp processes a chunk of the dot product
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            
            // Divide work among lanes in the warp
            for (int i = lane_id; i < in_features; i += WARP_SIZE) {
                float input_val = input[input_base + i];
                gate_sum += input_val * weights_gate[i * out_features + o];
                up_sum += input_val * weights_up[i * out_features + o];
            }
            
            // Warp-level reduction using shuffle
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
                up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
            }
            
            // Lane 0 has the final result
            if (lane_id == 0) {
                // Apply SwiGLU: swish(gate_sum) * up_sum
                output[idx] = swish(gate_sum) * up_sum;
            }
        }
    }
}

/**
 * Launch helper for the SwiGLU kernel
 */
extern "C" void launch_swiGLU(
    const float* input,
    const float* weights_gate,
    const float* weights_up,
    float* output,
    int batch_size,
    int seq_len,
    int in_features,
    int out_features,
    bool use_shuffle,
    cudaStream_t stream = 0
) {
    // Calculate grid and block dimensions
    int total_elements = batch_size * seq_len * out_features;
    int block_size = BLOCK_SIZE;  // 256 threads per block
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    // Cap grid size to prevent excessive launch
    if (grid_size > 65535) grid_size = 65535;
    
    // Launch appropriate kernel
    if (use_shuffle) {
        swiGLU_warp_shuffle<<<grid_size, block_size, 0, stream>>>(
            input, weights_gate, weights_up, output,
            batch_size, seq_len, in_features, out_features
        );
    } else {
        swiGLU_warp_optimized<<<grid_size, block_size, 0, stream>>>(
            input, weights_gate, weights_up, output,
            batch_size, seq_len, in_features, out_features
        );
    }
}

// Example host function to demonstrate usage
int main() {
    // Example dimensions
    int batch_size = 2;
    int seq_len = 128;
    int in_features = 768;
    int out_features = 768;
    
    // Calculate sizes
    size_t input_size = batch_size * seq_len * in_features * sizeof(float);
    size_t weights_size = in_features * out_features * sizeof(float);
    size_t output_size = batch_size * seq_len * out_features * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(input_size);
    float *h_weights_gate = (float*)malloc(weights_size);
    float *h_weights_up = (float*)malloc(weights_size);
    float *h_output = (float*)malloc(output_size);
    
    // Initialize input and weights (would normally load from somewhere)
    // Just filling with simple values for demonstration
    for (int i = 0; i < batch_size * seq_len * in_features; i++) {
        h_input[i] = (float)(i % 10) * 0.1f;
    }
    
    for (int i = 0; i < in_features * out_features; i++) {
        h_weights_gate[i] = (float)(i % 7) * 0.01f;
        h_weights_up[i] = (float)(i % 5) * 0.01f;
    }
    
    // Allocate device memory
    float *d_input, *d_weights_gate, *d_weights_up, *d_output;
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_weights_gate, weights_size);
    cudaMalloc((void**)&d_weights_up, weights_size);
    cudaMalloc((void**)&d_output, output_size);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_gate, h_weights_gate, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_up, h_weights_up, weights_size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    launch_swiGLU(
        d_input, d_weights_gate, d_weights_up, d_output,
        batch_size, seq_len, in_features, out_features,
        true  // Use shuffle-based implementation
    );
    
    // Copy results back
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // Verify results (simple check)
    printf("First few output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    
    // Cleanup
    free(h_input);
    free(h_weights_gate);
    free(h_weights_up);
    free(h_output);
    
    cudaFree(d_input);
    cudaFree(d_weights_gate);
    cudaFree(d_weights_up);
    cudaFree(d_output);
    
    return 0;
}