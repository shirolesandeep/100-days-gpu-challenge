import torch

def matrix_add():
    n = 32
    A = torch.ones(n, n, device='cuda')
    B = torch.full((n, n), 2.0, device='cuda')
    C = A + B
    print(f"Day 7 PyTorch: C[0,0] = {C[0,0].item()} (expected 3.0)")

if __name__ == "__main__":
    matrix_add()
