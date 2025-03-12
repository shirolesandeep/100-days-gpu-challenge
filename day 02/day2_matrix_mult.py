import torch

def matrix_mult():
    n = 32
    A = torch.ones(n, n, device='cuda')
    B = torch.full((n, n), 2.0, device='cuda')
    C = torch.matmul(A, B)
    print(f"Day 2 PyTorch: C[0,0] = {C[0,0].item()} (expected 64.0)")

if __name__ == "__main__":
    matrix_mult()
