import torch
import time

def matrix_mult_pytorch():
    # Matrix dimensions (same as CUDA example)
    M, N, K = 512, 512, 512

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    # Set device to CUDA
    device = torch.device("cuda")

    # Create random input matrices on CUDA device
    A = torch.rand(M, K, device=device, dtype=torch.float32)
    B = torch.rand(K, N, device=device, dtype=torch.float32)
    C = torch.zeros(M, N, device=device, dtype=torch.float32)

    # Warm-up run to initialize CUDA context
    _ = torch.matmul(A, B)

    # Synchronize to ensure accurate timing
    torch.cuda.synchronize()

    # Perform matrix multiplication
    start_time = time.time()
    C = torch.matmul(A, B)
    torch.cuda.synchronize()  # Wait for computation to complete
    end_time = time.time()

    print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds.")

    # Optional: Copy result back to CPU to verify (for profiling, keep on GPU)
    C_cpu = C.cpu()

if __name__ == "__main__":
    matrix_mult_pytorch()