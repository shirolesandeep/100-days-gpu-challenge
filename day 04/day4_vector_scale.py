import torch

def vector_scale():
    n = 1024
    x = torch.full((n,), 2.0, device='cuda')
    a = torch.tensor(5.0, device='cuda')
    y = a * x
    print(f"Day 4 PyTorch: y[0] = {y[0].item()} (expected 10.0)")

if __name__ == "__main__":
    vector_scale()
