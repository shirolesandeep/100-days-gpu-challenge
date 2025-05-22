import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(OptimizedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear layers for Q, K, V projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Scaling factor for attention
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project Q, K, V in one go and ensure contiguous memory
        qkv = self.qkv_proj(x).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # QK^T matmul with scaling
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Softmax in-place to reduce memory allocation
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        
        # Attention output: QK * V
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        return output

def main():
    # Example usage
    batch_size = 32
    seq_len = 128
    embed_dim = 256
    num_heads = 8
    
    # Create model and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedSelfAttention(embed_dim, num_heads).to(device)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Run forward pass
    output = model(x)
    
    # Print output shape
    print(f"Output shape: {output.shape}")
    
    # Optional: Profile memory usage
    if torch.cuda.is_available():
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB")

if __name__ == "__main__":
    main()