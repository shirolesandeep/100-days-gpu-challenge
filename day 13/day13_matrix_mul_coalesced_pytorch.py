import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Matrix size
N = 512
TILE_SIZE = 32

# Create input matrices (ensure contiguous memory layout)
A = torch.arange(N * N, dtype=torch.float32).reshape(N, N) % 10
B = (torch.arange(N * N, dtype=torch.float32) + 1).reshape(N, N) % 10
A = A.contiguous().to(device)  # Ensure contiguous memory
B = B.contiguous().to(device)

# Tiled matrix multiplication with coalesced access
def tiled_matmul_coalesced(A, B, tile_size):
    N = A.shape[0]
    C = torch.zeros(N, N, device=device)
    
    start_time = time.time()
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, N, tile_size):
                i_end = min(i + tile_size, N)
                j_end = min(j + tile_size, N)
                k_end = min(k + tile_size, N)
                
                # Access tiles in a coalesced manner (contiguous memory)
                tile_A = A[i:i_end, k:k_end].contiguous()
                tile_B = B[k:k_end, j:j_end].contiguous()
                C[i:i_end, j:j_end] += torch.matmul(tile_A, tile_B)
    
    torch.cuda.synchronize()
    end_time = time.time()
    return C, end_time - start_time

# Run computation
C, exec_time = tiled_matmul_coalesced(A, B, TILE_SIZE)

# Move to CPU for printing
C_cpu = C.cpu()

# Print sample results
print("Sample results (top-left 3x3):")
print(C_cpu[:3, :3])

# Print execution time
print(f"Execution time: {exec_time:.4f} seconds")
print(f"Result tensor is on: {C.device}")
