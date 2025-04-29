import torch
import torch.nn as nn
import torch.cuda.amp as amp
import time
import math
import uuid

# FP8 emulation class
class FP8:
    def __init__(self, value):
        self.value = self._to_fp8(value)

    def _to_fp8(self, x):
        if x == 0:
            return 0
        sign = -1 if x < 0 else 1
        x = abs(x)
        exponent = math.floor(math.log2(x)) if x != 0 else -7
        mantissa = x / (2 ** exponent) - 1.0
        exponent = max(-7, min(8, exponent + 7))
        mantissa = max(0, min(7, int(mantissa * 8)))
        return (sign < 0) << 7 | (exponent & 0x0F) << 3 | (mantissa & 0x07)

    def to_float(self):
        if self.value == 0:
            return 0.0
        sign = -1 if (self.value >> 7) else 1
        exponent = (self.value >> 3) & 0x0F
        mantissa = self.value & 0x07
        return sign * (1.0 + mantissa / 8.0) * (2 ** (exponent - 7))

# Custom FP8 tensor wrapper
class FP8Tensor:
    def __init__(self, tensor):
        self.data = [FP8(x.item()) for x in tensor.flatten()]
        self.shape = tensor.shape
        self.device = tensor.device

    def to(self, device):
        self.device = device
        return self

    def to_float(self):
        return torch.tensor([x.to_float() for x in self.data], device=self.device).reshape(self.shape)

# Transformer block module
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

def benchmark_transformer(dtype, batch, seq_len, d_model, n_heads, iterations, device):
    d_ff = 4 * d_model
    model = TransformerBlock(d_model, n_heads, d_ff).to(device)
    input = torch.rand(seq_len, batch, d_model, device=device)
    
    if dtype == torch.float32:
        model = model.float()
        input = input.float()
    elif dtype == torch.float16:
        model = model.half()
        input = input.half()
    else:  # FP8
        input = FP8Tensor(input)
        input = input.to_float()  # Convert back for computation
    
    # Warm-up
    for _ in range(5):
        output = model(input)
        loss = output.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        if dtype == torch.float16:
            with amp.autocast():
                output = model(input)
                loss = output.sum()
            loss.backward()
        else:
            output = model(input)
            loss = output.sum()
            loss.backward()
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / iterations
    
    return elapsed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, seq_len, d_model, n_heads = 32, 128, 512, 8
    iterations = 100

    print(f"Running benchmark on {device}")
    
    # Benchmark FP32
    time_fp32 = benchmark_transformer(torch.float32, batch, seq_len, d_model, n_heads, iterations, device)
    print(f"FP32 Transformer Block: {time_fp32:.3f} ms per iteration")
    
    # Benchmark FP16
    time_fp16 = benchmark_transformer(torch.float16, batch, seq_len, d_model, n_heads, iterations, device)
    print(f"FP16 Transformer Block: {time_fp16:.3f} ms per iteration")
    
    # Benchmark FP8 (emulated)
    time_fp8 = benchmark_transformer(None, batch, seq_len, d_model, n_heads, iterations, device)
    print(f"FP8 Transformer Block: {time_fp8:.3f} ms per iteration")

if __name__ == "__main__":
    main()