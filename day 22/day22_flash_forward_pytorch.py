import torch
import torch.nn.functional as F

def flash_attention_forward(Q, K, V):
    N, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [N, N]
    P = F.softmax(S, dim=-1)                          # [N, N]
    O = torch.matmul(P, V)                            # [N, d]
    return O

# Example usage
if __name__ == "__main__":
    N, d = 64, 32
    Q = torch.randn(N, d).cuda()
    K = torch.randn(N, d).cuda()
    V = torch.randn(N, d).cuda()
    O = flash_attention_forward(Q, K, V)
    print(O.shape)  # Should be [N, d]
