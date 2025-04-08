import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

# Example usage
model = FeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 64, 512)
output = model(x)
