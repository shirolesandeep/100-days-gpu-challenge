import torch
import torch.nn.functional as F

def tiled_2d_conv(input, kernel, tile_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    kernel = kernel.to(device)

    height, width = input.shape
    k_size = kernel.shape[0]
    output = torch.zeros(height, width, device=device)

    # Pad input
    padding = k_size // 2
    padded_input = F.pad(input.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='constant', value=0)

    # Tile processing
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            i_end = min(i + tile_size + k_size - 1, height + 2 * padding)
            j_end = min(j + tile_size + k_size - 1, width + 2 * padding)
            tile = padded_input[:, :, i:i_end, j:j_end]
            tile_output = F.conv2d(tile, kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze()
            output[i:min(i + tile_size, height), j:min(j + tile_size, width)] = \
                tile_output[:min(tile_size, height - i), :min(tile_size, width - j)]

    return output.cpu()

# Example usage
input = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=torch.float32)
kernel = torch.tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=torch.float32)

output = tiled_2d_conv(input, kernel)
print("2D Convolution output:")
print(output)