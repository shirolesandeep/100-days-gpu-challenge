import torch
import math

def bitonic_sort(arr):
    """
    Perform bitonic sort on a 1D PyTorch tensor.
    Input tensor size must be a power of 2.
    """
    n = arr.size(0)
    assert (n & (n - 1)) == 0, "Input size must be a power of 2"
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = arr.to(device)
    
    stages = int(math.log2(n))
    
    for stage in range(stages):
        for substage in range(stage + 1):
            pair_distance = 1 << (stage - substage)
            block_width = 2 * pair_distance
            
            # Create indices for comparison
            idx = torch.arange(n, device=device)
            left_id = idx - (idx % block_width) + (idx % pair_distance)
            right_id = left_id + pair_distance
            
            # Create mask for valid comparisons
            mask = right_id < n
            
            # Determine direction (ascending or descending)
            direction = ((idx // (1 << stage)) % 2 == 0).float() * 2 - 1
            
            # Get left and right elements
            left = arr[left_id]
            right = arr[right_id]
            
            # Compare and swap based on direction
            swap = torch.zeros(n, device=device, dtype=torch.bool)
            swap[mask] = (direction[mask] == 1) & (left[mask] > right[mask]) | \
                        (direction[mask] == -1) & (left[mask] < right[mask])
            
            # Perform swap
            arr_temp = arr.clone()
            arr[left_id[swap]] = right[swap]
            arr[right_id[swap]] = left[swap]
    
    return arr.cpu()

def main():
    # Test the bitonic sort
    n = 1024  # Must be power of 2
    # Generate random input
    arr = torch.randint(0, 1000, (n,), dtype=torch.int32)
    
    print("First 10 unsorted elements:")
    print(arr[:10].numpy())
    
    # Perform sort
    sorted_arr = bitonic_sort(arr)
    
    print("\nFirst 10 sorted elements:")
    print(sorted_arr[:10].numpy())
    
    # Verify sort
    is_sorted = torch.all(sorted_arr[:-1] <= sorted_arr[1:])
    print(f"\nArray is {'sorted' if is_sorted else 'not sorted'}")

if __name__ == "__main__":
    main()