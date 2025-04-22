import torch
import torch.nn as nn
import torch.cuda.amp as amp

# Custom 2D Convolution Layer
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2dLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)

def main():
    # Hyperparameters
    width = 64
    height = 64
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    batch_size = 1

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Tensor Core usage.")

    # Initialize model
    model = Conv2dLayer(in_channels, out_channels, kernel_size).to(device).half()

    # Initialize input and kernel
    input_tensor = torch.ones(batch_size, in_channels, height, width, device=device, dtype=torch.float16)
    with torch.no_grad():
        model.conv.weight.fill_(0.1111)  # Simple averaging kernel (1/9)

    # Mixed precision context
    scaler = amp.GradScaler()

    # Forward pass with mixed precision
    with amp.autocast():
        output = model(input_tensor)

    # Print sample output
    print(f"Sample output[0,0,0,0]: {output[0,0,0,0].item():.4f}")

    # Optional: Verify output shape
    expected_out_size = width - kernel_size + 1
    assert output.shape == (batch_size, out_channels, expected_out_size, expected_out_size), \
        f"Unexpected output shape: {output.shape}"

if __name__ == "__main__":
    main()