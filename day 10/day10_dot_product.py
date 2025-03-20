import torch

def dot_product():
    n = 1024
    x = torch.full((n,), 2.0, device='cuda')
    y = torch.full((n,), 3.0, device='cuda')
    dot = torch.dot(x, y)
    print(f"Day 10 PyTorch: Dot product = {dot.item()} (expected 6144.0)")

if __name__ == "__main__":
    dot_product()
