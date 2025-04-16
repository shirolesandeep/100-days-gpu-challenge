import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load

# Load the custom CUDA kernel
tensor_core_ops = load(
    name="tensor_core_ops",
    sources=["transformer_tensor_core.cu"],
    verbose=True,
    extra_cuda_cflags=["-arch=sm_80", "--ptxas-options=-v", "-use_fast_math"]
)

class TensorCoreLinear(nn.Module):
    """Linear layer that uses Tensor Cores for matrix multiplication"""
    def __init__(self, in_features, out_features, bias=True):
        super(TensorCoreLinear, self).__init__()
        # Weight must be in FP16 for Tensor Core operations
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Ensure input is in FP16
        if x.dtype != torch.float16:
            x = x.half()
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        embedding_dim = x.size(2)
        
        # Allocate output tensor
        output = torch.empty(batch_size, seq_len, self.weight.size(0), 
                            dtype=torch.float16, device=x.device)
        
        # Use our custom tensor core kernel
        tensor_core_ops.launch_tensor_core_matmul(
            x.contiguous(),
            self.weight.contiguous(),
            output,
            batch_size,
            seq_len,
            embedding_dim,
            torch.cuda.current_stream().cuda_stream
        )
        
        # Add bias if present
        if self.bias is not None:
            output += self.bias
            
        return output

class TensorCoreMultiHeadAttention(nn.Module):
    """Multi-head attention implemented with Tensor Cores"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(TensorCoreMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Create projection matrices in FP16
        self.q_proj = TensorCoreLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = TensorCoreLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = TensorCoreLinear(embed_dim, embed_dim, bias=False)
        self.out_proj = TensorCoreLinear(embed_dim, embed_dim)
        
        self.dropout = dropout
    
    def forward(self, query, key, value, attn_mask=None):
        # Convert inputs to FP16 if needed
        if query.dtype != torch.float16:
            query = query.half()
            key = key.half()
            value = value.half()
        
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multihead attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Allocate output tensor
        output = torch.empty_like(q)
        
        # For each head, call the tensor core attention
        for h in range(self.num_heads):
            tensor_core_ops.launch_tensor_core_self_attention(
                q[:, h].contiguous(),
                k[:, h].contiguous(),
                v[:, h].contiguous(),
                output[:, h].contiguous(),
                batch_size,
                1,  # single head at a time
                seq_len,
                self.head_dim,
                self.scale,
                torch.cuda.current_stream().cuda_stream
            )
        
        # Reshape output and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output

class TensorCoreFFN(nn.Module):
    """Feed-forward network using Tensor Cores"""
    def __init__(self, embed_dim, ffn_dim, dropout=0.0):
        super(TensorCoreFFN, self).__init__()
        self.fc1 = TensorCoreLinear(embed_dim, ffn_dim)
        self.fc2 = TensorCoreLinear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store dimensions for CUDA kernel
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
    
    def forward(self, x):
        # Convert to FP16 if needed
        if x.dtype != torch.float16:
            x = x.half()
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Allocate output tensor
        output = torch.empty_like(x)
        
        # Call tensor core FFN kernel
        tensor_core_ops.launch_tensor_core_ffn(
            x.contiguous(),
            self.fc1.weight.contiguous(),
            self.fc2.weight.contiguous(),
            output,
            batch_size,
            seq_len,
            self.embed_dim,
            self.ffn_dim,
            torch.cuda.current_stream().cuda_stream
        )
        
        # In a real implementation, we'd need to handle biases and dropout here
        # For simplicity, we're just using the kernel output
        
        return output

class TensorCoreTransformerLayer(nn.Module):
    """Transformer layer with Tensor Core optimizations"""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super(TensorCoreTransformerLayer, self).__init__()
        
        # Self-attention with tensor cores
        self.self_attn = TensorCoreMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer norm (standard PyTorch)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # Feed-forward network with tensor cores
        self.ffn = TensorCoreFFN(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout
        )
        
        # Dropout (standard PyTorch)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, self_attn_mask=None):
        # Convert to FP16 for tensor core operations
        if x.dtype != torch.float16:
            x = x.half()
        
        # Self-attention with residual connection and layer norm
        residual = x
        x = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = self.dropout(x)
        x = residual + x
        x = self.norm1(x.float()).half()  # LayerNorm in fp32, then back to fp16
        
        # FFN with residual connection and layer norm
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        x = self.norm2(x.float()).half()  # LayerNorm in fp32, then back to fp16
        
        return x

# Usage example
def test_tensor_core_transformer():
    # Initialize model
    model = TensorCoreTransformerLayer(
        embed_dim=768,  # Embedding dimension
        num_heads=12,   # Number of attention heads
        ffn_dim=3072,   # Feed-forward dimension
        dropout=0.1     # Dropout rate
    ).cuda().half()     # Move to GPU and convert to FP16
    
    # Test with a sample input
    batch_size = 8
    seq_len = 512
    embed_dim = 768
    
    # Create a sample input tensor
    x = torch.randn(batch_size, seq_len, embed_dim, device='cuda', dtype=torch.float16)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Successfully ran tensor core optimized transformer layer!")

if __name__ == "__main__":
    test_tensor_core_transformer()