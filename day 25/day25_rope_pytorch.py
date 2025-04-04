import torch

def apply_rope(Q, K, pos_offset=0):
    N, d = Q.shape
    device = Q.device
    positions = torch.arange(pos_offset, pos_offset + N, device=device).float()
    theta = 10000.0 ** (-2.0 * torch.arange(0, d, 2, device=device) / d)
    angles = positions[:, None] * theta[None, :]  # [N, d//2]
    
    cos_vals = torch.cos(angles).repeat(1, 2)      # [N, d]
    sin_vals = torch.sin(angles).repeat(1, 2)      # [N, d]
    
    Q_rolled = torch.roll(Q, shifts=1, dims=-1)    # Pairwise rotation
    K_rolled = torch.roll(K, shifts=1, dims=-1)
    
    Q_rot = Q * cos_vals - Q_rolled * sin_vals
    K_rot = K * cos_vals - K_rolled * sin_vals
    return Q_rot, K_rot

# Example usage
if __name__ == "__main__":
    N, d = 64, 32
    Q = torch.randn(N, d).cuda()
    K = torch.randn(N, d).cuda()
    Q_rot, K_rot = apply_rope(Q, K)
    print(Q_rot.shape, K_rot.shape)  # Should both be [N, d]