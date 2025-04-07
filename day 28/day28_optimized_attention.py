import torch
from torch import nn

class OptimizedAttention(nn.Module):
    def __init__(self, d_model, n_heads, tile_size=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.tile_size = tile_size

    def forward(self, Q, K, V):
        batch_size, seq_len, _ = Q.size()
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Tiled computation for large sequences
        scores = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=Q.device)
        for i in range(0, seq_len, self.tile_size):
            end = min(i + self.tile_size, seq_len)
            scores[:, :, :, i:end] = torch.matmul(Q, K[:, :, i:end].transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

# Example usage with memory profiling
model = OptimizedAttention(d_model=512, n_heads=8)
Q, K, V = torch.randn(2, 1024, 512).cuda(), torch.randn(2, 1024, 512).cuda(), torch.randn(2, 1024, 512).cuda()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    output = model(Q, K, V)
print(prof.key_averages().table(sort_by="cuda_memory_usage"))
