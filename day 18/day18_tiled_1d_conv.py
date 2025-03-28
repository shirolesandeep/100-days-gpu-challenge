import torch
import torch.nn.functional as F

def tiled_1d_conv(input, kernel, tile_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    kernel = kernel.to(device)

    input_len = input.shape[0]
    kernel_len = kernel.shape[0]
    output_len = input_len
    output = torch.zeros(output_len, device=device)

    # Pad input
    padding = kernel_len // 2
    padded_input = F.pad(input.unsqueeze(0).unsqueeze(0), (padding, padding), mode='constant', value=0)

    # Tile processing
    for start in range(0, input_len, tile_size):
        end = min(start + tile_size + kernel_len - 1, input_len + 2 * padding)
        tile = padded_input[:, :, start:end]
        tile_output = F.conv1d(tile, kernel.view(1, 1, -1), padding=0).squeeze()
        output[start:start + tile_size] = tile_output[:min(tile_size, input_len - start)]

    return output.cpu()

# Example usage
input = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
kernel = torch.tensor([1, 0, -1], dtype=torch.float32)  # Edge detection

output = tiled_1d_conv(input, kernel)
print("Convolution output:")
print(output)
