import torch

def flash_attention_backward(dO, Q, K, V, P):
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    dS = torch.matmul(dO, V.transpose(-2, -1)) * P  # [N, N]
    dQ = torch.matmul(dS, K) * scale                # [N, d]
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale  # [N, d]
    dV = torch.matmul(P.transpose(-2, -1), dO)      # [N, d]
    return dQ, dK, dV

# Example usage
if __name__ == "__main__":
    N, d = 64, 32
    Q = torch.randn(N, d).cuda()
    K = torch.randn(N, d).cuda()
    V = torch.randn(N, d).cuda()
    dO = torch.randn(N, d).cuda()
    O = flash_attention_forward(Q, K, V)
    P = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / (d ** 0.5)), dim=-1)
    dQ, dK, dV = flash_attention_backward(dO, Q, K, V, P)
    print(dQ.shape, dK.shape, dV.shape)  # Should all be [N, d]
