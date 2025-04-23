import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Define the CUDA kernel
cuda_source = """
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N,
    int D,
    float eps
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / D;
    int feat_idx = idx % D;
    
    if (idx >= N * D) return;
    
    // Shared memory for partial sums
    __shared__ float shared_mean[32];
    __shared__ float shared_var[32];
    
    // Compute mean
    float sum = 0.0f;
    if (feat_idx < D) {
        sum = input[batch_idx * D + feat_idx];
    }
    
    sum = warp_reduce_sum(sum);
    
    if (warp.thread_rank() == 0) {
        shared_mean[threadIdx.x / 32] = sum / D;
    }
    block.sync();
    
    float mean = shared_mean[threadIdx.x / 32];
    
    // Compute variance
    float var_sum = 0.0f;
    if (feat_idx < D) {
        float diff = input[batch_idx * D + feat_idx] - mean;
        var_sum = diff * diff;
    }
    
    var_sum = warp_reduce_sum(var_sum);
    
    if (warp.thread_rank() == 0) {
        shared_var[threadIdx.x / 32] = var_sum / D;
    }
    block.sync();
    
    float var = shared_var[threadIdx.x / 32];
    float std = sqrtf(var + eps);
    
    // Normalize and apply affine transform
    if (feat_idx < D) {
        float x = input[batch_idx * D + feat_idx];
        output[batch_idx * D + feat_idx] = 
            ((x - mean) / std) * gamma[feat_idx] + beta[feat_idx];
    }
}

// Wrapper function to launch the kernel
torch::Tensor layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    const auto N = input.size(0);
    const auto D = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (N * D + threads - 1) / threads;
    
    layer_norm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        N,
        D,
        eps
    );
    
    return output;
}
"""

# Compile the CUDA extension
custom_layer_norm = load(
    name='custom_layer_norm',
    sources=[os.path.join(os.path.dirname(__file__), 'custom_layer_norm_cuda.cpp')],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("CustomLayerNorm only supports CUDA tensors")
        if x.dtype != torch.float32:
            raise RuntimeError("CustomLayerNorm only supports float32")
            
        return custom_layer_norm.layer_norm_cuda(x, self.gamma, self.beta, self.eps)

# Example usage
if __name__ == "__main__":
    # Create sample input
    batch_size, feature_dim = 32, 512
    x = torch.randn(batch_size, feature_dim).cuda()
    
    # Initialize layer norm
    layer_norm = CustomLayerNorm(feature_dim).cuda()
    
    # Forward pass
    output = layer_norm(x)
    
    # Verify with PyTorch's LayerNorm
    torch_layer_norm = nn.LayerNorm(feature_dim).cuda()
    torch_layer_norm.weight = layer_norm.gamma
    torch_layer_norm.bias = layer_norm.beta
    torch_output = torch_layer_norm(x)
    
    # Check if results are close
    assert torch.allclose(output, torch_output, atol=1e-4), "Results don't match"
    print("Custom LayerNorm matches PyTorch's LayerNorm!")