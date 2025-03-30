import torch
import time

def profile_matrix_mul(A, B, runs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    B = B.to(device)

    # Warm-up
    C = torch.matmul(A, B)

    # Profile
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(runs):
        C = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event) / runs
    print(f"PyTorch Matrix Mul ({A.shape[0]}x{A.shape[1]} x {B.shape[0]}x{B.shape[1]}): {elapsed_time_ms:.3f} ms (avg over {runs} runs)")
    
    return C.cpu()

# Example usage
m, n, k = 1024, 1024, 1024
A = torch.rand(m, k, dtype=torch.float32)
B = torch.rand(k, n, dtype=torch.float32)

C = profile_matrix_mul(A, B)
