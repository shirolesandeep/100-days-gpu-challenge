import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Matrix size and sparsity
N = 512
DENSITY = 0.1

# Create sparse matrix in CSR format
nnz = int(N * N * DENSITY)  # Number of non-zero elements
values = []
col_indices = []
row_ptr = [0]
row_count = 0
idx = 0

for i in range(N):
    non_zeros_in_row = 0
    for j in range(N):
        if (i + j) % 10 == 0 and idx < nnz:
            values.append(float(i + j + 1))
            col_indices.append(j)
            idx += 1
            non_zeros_in_row += 1
    row_count += non_zeros_in_row
    row_ptr.append(row_count)

# Convert to tensors
values = torch.tensor(values, dtype=torch.float32)
col_indices = torch.tensor(col_indices, dtype=torch.int64)
row_ptr = torch.tensor(row_ptr, dtype=torch.int64)

# Create sparse CSR tensor
A = torch.sparse_csr_tensor(row_ptr, col_indices, values, size=(N, N), dtype=torch.float32, device=device)

# Create dense vector x
x = torch.arange(N, dtype=torch.float32) % 10
x = x.to(device)

# Perform SpMV
start_time = time.time()
y = A @ x  # Matrix-vector multiplication
torch.cuda.synchronize()
end_time = time.time()

# Print execution time
print(f"Execution time: {(end_time - start_time) * 1000:.4f} ms")

# Print first few elements of result
y_cpu = y.cpu()
print("First 5 elements of result vector y:")
print(y_cpu[:5])
