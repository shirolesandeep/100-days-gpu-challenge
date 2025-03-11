import torch

def vector_add():
    n = 1024
    x = torch.ones(n, device='cuda')
    y = torch.full((n,), 2.0, device='cuda')
    z = x + y
    print(f"Day 1 PyTorch: z[0] = {z[0].item()} (expected 3.0)")

if __name__ == "__main__":
    vector_add()