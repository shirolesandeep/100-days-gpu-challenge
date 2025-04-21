import torch
import math

# Configuration
THRESHOLD = 1024
BLOCK_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def recursive_sum(input_tensor, n, level=0):
    """
    Recursively compute the sum of input_tensor of size n.
    Args:
        input_tensor: Input tensor on GPU
        n: Size of the input tensor or segment
        level: Recursion depth
    Returns:
        Scalar sum of the input tensor
    """
    # Base case: if segment size is small, compute sum directly
    if n <= THRESHOLD:
        return torch.sum(input_tensor[:n]).item()

    # Emulate CUDA grid size
    grid_size = math.ceil(n / BLOCK_SIZE)
    segment_size = math.ceil(n / grid_size)

    # Initialize output tensor for partial sums
    output = torch.zeros(grid_size, dtype=torch.int32, device=DEVICE)

    # Process each segment
    for i in range(grid_size):
        start = i * segment_size
        end = min(start + segment_size, n)
        if start < n:
            segment = input_tensor[start:end]
            # Perform reduction on segment
            output[i] = torch.sum(segment)

    # Recursively sum the partial sums
    return recursive_sum(output, grid_size, level + 1)

def main():
    # Array size (power of 2 for simplicity)
    n = 1 << 20  # 1M elements

    # Initialize input array
    h_input = torch.ones(n, dtype=torch.int32)
    expected_sum = n  # Since each element is 1

    # Move to device
    d_input = h_input.to(DEVICE)

    try:
        # Compute sum
        result = recursive_sum(d_input, n)
        
        # Print result
        print(f"Sum of array: {result}")
        
        # Verify result
        if result == expected_sum:
            print("Result is correct!")
        else:
            print(f"Result is incorrect! Expected {expected_sum}")

    except RuntimeError as e:
        print(f"PyTorch error: {e}")
        return

if __name__ == "__main__":
    main()