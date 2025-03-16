import torch
def vector_sub():
    n = 1024
    x = torch.full((n,), 5.0, device='cuda')
    y = torch.full((n,), 3.0, device='cuda')
    z = x - y
    print(f"Day 6 PyTorch: z[0] = {z[0].item()} (expected 2.0)")

if __name__ == "__main__":
    vector_sub()