import torch

def reduce_sum():
    n = 1024
    x = torch.ones(n, device='cuda')
    sum_val = torch.sum(x)
    print(f"Day 5 PyTorch: Sum = {sum_val.item()} (expected 1024.0)")

if __name__ == "__main__":
    reduce_sum()
