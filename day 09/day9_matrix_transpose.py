import torch

def matrix_transpose():
    n = 32
    A = torch.arange(1, n * n + 1, dtype=torch.float32, device='cuda').reshape(n, n)
    B = A.transpose(0, 1)
    print(f"Day 9 PyTorch: B[0,1] = {B[0,1].item()} (expected 2.0)")

if __name__ == "__main__":
    matrix_transpose()