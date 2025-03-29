%%writefile day11_matrix_mul_pytorch.py
import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Matrix size
N = 512

# Create input matrices
A = torch.arange(N * N, dtype=torch.float32).reshape(N, N) % 10  # Simple pattern
B = (torch.arange(N * N, dtype=torch.float32) + 1).reshape(N, N) % 10
A = A.to(device)
B = B.to(device)

# Perform matrix multiplication
start_time = time.time()
C = torch.matmul(A, B)
torch.cuda.synchronize()  # Wait for GPU to finish
end_time = time.time()

# Move to CPU for printing
C_cpu = C.cpu()

# Print sample results
print("Sample results (top-left 3x3):")
print(C_cpu[:3, :3])

# Print execution time
print(f"Execution time: {end_time - start_time:.4f} seconds")
print(f"Result tensor is on: {C.device}")
