import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import math

# Problem dimensions
SEQ_LEN = 512
HEAD_DIM = 64
NUM_HEADS = 16
BATCH_SIZE = 1
TILE_SIZE = 64  # Tile size for memory efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16  # Use half-precision for memory savings

def flash_attention_forward(Q, K, V, seq_len, head_dim, tile_size):
    """
    Simplified FlashAttention forward pass with tiling.
    Args:
        Q: Queries (batch_size, num_heads, seq_len, head_dim)
        K: Keys (batch_size, num_heads, seq_len, head_dim)
        V: Values (batch_size, num_heads, seq_len, head_dim)
        seq_len: Sequence length
        head_dim: Head dimension
        tile_size: Size of tiles for Q, K, V
    Returns:
        O: Output tensor (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, _, _ = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    O = torch.zeros_like(Q)  # Output tensor

    # Iterate over tiles
    for i in range(0, seq_len, tile_size):
        # Load Q tile
        Q_tile = Q[:, :, i:i+tile_size, :]  # (batch_size, num_heads, tile_size, head_dim)
        for j in range(0, seq_len, tile_size):
            with record_function("Load_KV_Tile"):
                # Load K and V tiles
                K_tile = K[:, :, j:j+tile_size, :]  # (batch_size, num_heads, tile_size, head_dim)
                V_tile = V[:, :, j:j+tile_size, :]  # (batch_size, num_heads, tile_size, head_dim)

            with record_function("Compute_Scores"):
                # Compute attention scores S = QK^T / sqrt(d)
                S = torch.einsum("bhid,bhjd->bhij", Q_tile, K_tile) * scale  # (batch_size, num_heads, tile_size, tile_size)

            with record_function("Softmax"):
                # Softmax normalization (on-the-fly)
                S_max = S.max(dim=-1, keepdim=True)[0]
                P = torch.exp(S - S_max)  # Subtract max for numerical stability
                P = P / P.sum(dim=-1, keepdim=True)  # (batch_size, num_heads, tile_size, tile_size)

            with record_function("Compute_Output"):
                # Compute output O = PV
                O[:, :, i:i+tile_size, :] += torch.einsum("bhij,bhjd->bhid", P, V_tile)

    return O

def main():
    # Initialize input tensors
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # Warm-up run to stabilize CUDA
    _ = flash_attention_forward(Q, K, V, SEQ_LEN, HEAD_DIM, TILE_SIZE)
    torch.cuda.synchronize()

    # Profile the forward pass
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("FlashAttention_Forward"):
            O = flash_attention_forward(Q, K, V, SEQ_LEN, HEAD_DIM, TILE_SIZE)
        torch.cuda.synchronize()

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    print(f"Peak CUDA memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    # Verify output (basic sanity check)
    print(f"Output shape: {O.shape}")
    print(f"Output mean: {O.mean().item():.4f}")

if __name__ == "__main__":
    main()