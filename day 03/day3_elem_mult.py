import torch

def elem_mult():
    n = 1024
    x = torch.full((n,), 2.0, device='cuda')
    y = torch.full((n,), 3.0, device='cuda')
    z = x * y
    print(f"Day 3 PyTorch: z[0] = {z[0].item()} (expected 6.0)")

if __name__ == "__main__":
    elem_mult()
