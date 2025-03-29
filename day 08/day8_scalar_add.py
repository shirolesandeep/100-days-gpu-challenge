import torch

def scalar_add():
    n = 1024
    x = torch.full((n,), 2.0, device='cuda')
    a = torch.tensor(3.0, device='cuda')
    y = x + a
    print(f"Day 8 PyTorch: y[0] = {y[0].item()} (expected 5.0)")

if __name__ == "__main__":
    scalar_add()
