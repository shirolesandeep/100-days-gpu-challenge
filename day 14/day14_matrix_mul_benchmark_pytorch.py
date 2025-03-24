import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Matrix size
N = 512
TILE_SIZE = 32
NUM_RUNS = 10

# Create input matrices
A = torch.arange(N * N, dtype=torch.float32).reshape(N, N) % 10
B = (torch.arange(N * N, dtype=torch.float32) + 1).reshape(N, N) % 10
A = A.contiguous().to(device)
B = B.contiguous().to(device)

# Tiled matrix multiplication (Day 12)
def tiled_matmul(A, B, tile_size):
    N = A.shape[0]
    C = torch.zeros(N, N, device=device)
    
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, N, tile_size):
                i_end = min(i + tile_size, N)
                j_end = min(j + tile_size, N)
                k_end = min(k + tile_size, N)
                
                tile_A = A[i:i_end, k:k_end]
                tile_B = B[k:k_end, j:j_end]
                C[i:i_end, j:j_end] += torch.matmul(tile_A, tile_B)
    
    return C

# Coalesced matrix multiplication (Day 13)
def tiled_matmul_coalesced(A, B, tile_size):
    N = A.shape[0]
    C = torch.zeros(N, N, device=device)
    
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, N, tile_size):
                i_end = min(i + tile_size, N)
                j_end = min(j + tile_size, N)
                k_end = min(k + tile_size, N)
                
                tile_A = A[i:i_end, k:k_end].contiguous()
                tile_B = B[k:k_end, j:j_end].contiguous()
                C[i:i_end, j:j_end] += torch.matmul(tile_A, tile_B)
    
    return C

# Benchmark PyTorch matmul
pytorch_time = 0
for i in range(NUM_RUNS):
    start_time = time.time()
    C_pytorch = torch.matmul(A, B)
    torch.cuda.synchronize()
    end_time = time.time()
    pytorch_time += (end_time - start_time) * 1000  # Convert to ms
print(f"PyTorch matmul Average Time: {pytorch_time / NUM_RUNS:.4f} ms")

# Benchmark Tiled (Day 12)
tiled_time = 0
for i in range(NUM_RUNS):
    start_time = time.time()
    C_tiled = tiled_matmul(A, B, TILE_SIZE)
    torch.cuda.synchronize()
    end_time = time.time()
    tiled_time += (end_time - start_time) * 1000
print(f"Tiled Matrix Mul (Day 12) Average Time: {tiled_time / NUM_RUNS:.4f} ms")

# Benchmark Coalesced (Day 13)
coalesced_time = 0
for i in range(NUM_RUNS):
    start_time = time.time()
    C_coalesced = tiled_matmul_coalesced(A, B, TILE_SIZE)
    torch.cuda.synchronize()
    end_time = time.time()
    coalesced_time += (end_time - start_time) * 1000
print(f"Coalesced Matrix Mul (Day 13) Average Time: {coalesced_time / NUM_RUNS:.4f} ms")

# Print sample results (using PyTorch result)
C_cpu = C_pytorch.cpu()
print("Sample results (top-left 3x3):")
print(C_cpu[:3, :3])
