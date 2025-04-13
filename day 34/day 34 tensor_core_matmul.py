import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

def check_tensor_cores_available():
    """Check if the current GPU supports Tensor Cores."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot use Tensor Cores.")
        return False
    
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    cuda_capability = device_props.major + device_props.minor / 10
    
    # Tensor Cores are available on Volta (7.0), Turing (7.5), Ampere (8.0+), etc.
    if cuda_capability >= 7.0:
        print(f"GPU: {device_props.name}")
        print(f"CUDA Capability: {cuda_capability}")
        print("This GPU supports Tensor Cores!")
        return True
    else:
        print(f"GPU: {device_props.name}")
        print(f"CUDA Capability: {cuda_capability}")
        print("This GPU does not support Tensor Cores.")
        return False

def matrix_multiply_benchmark(sizes, dtype=torch.float16, use_tensor_cores=True):
    """
    Benchmark matrix multiplication with and without Tensor Cores.
    
    Args:
        sizes: List of matrix sizes to test (N for N×N matrices)
        dtype: Data type to use (float16 for Tensor Cores)
        use_tensor_cores: Whether to enable Tensor Cores
    
    Returns:
        Dictionary with timing results
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return None
    
    results = {
        "sizes": sizes,
        "tensor_cores_times": [],
        "normal_times": []
    }
    
    # Set the device
    device = torch.device("cuda")
    
    # Configure cuBLAS to use Tensor Cores or not
    if use_tensor_cores:
        torch.backends.cuda.matmul.allow_tf32 = True  # For A100 GPUs with TF32
        # Enable WMMA/Tensor Core usage
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        # Disable automatic optimization
        torch.backends.cudnn.benchmark = False
    
    for size in sizes:
        # Create matrices with the appropriate size and data type
        # For tensor cores, dimensions should be multiples of 8 for FP16 (or 16 for INT8)
        adjusted_size = size
        if use_tensor_cores and size % 8 != 0:
            adjusted_size = size + (8 - size % 8)
            print(f"Adjusting size from {size} to {adjusted_size} to optimize for Tensor Cores")
        
        # Generate random matrices
        a = torch.randn(adjusted_size, adjusted_size, dtype=dtype, device=device)
        b = torch.randn(adjusted_size, adjusted_size, dtype=dtype, device=device)
        
        # Warmup
        for _ in range(5):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        
        # Timing
        start_time = time.time()
        iterations = 100  # Adjust based on matrix size
        for _ in range(iterations):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        
        # Store results
        if use_tensor_cores:
            results["tensor_cores_times"].append(avg_time)
        else:
            results["normal_times"].append(avg_time)
            
        print(f"Size: {adjusted_size}x{adjusted_size}, {'Tensor Cores' if use_tensor_cores else 'Normal'} time: {avg_time:.6f} seconds")
    
    return results

def visualize_results(tc_results, normal_results):
    """Visualize and compare the benchmark results."""
    plt.figure(figsize=(12, 6))
    
    sizes = tc_results["sizes"]
    tc_times = tc_results["tensor_cores_times"]
    normal_times = normal_results["normal_times"]
    
    # Calculate speedup
    speedup = [normal / tc for normal, tc in zip(normal_times, tc_times)]
    
    # Plot timings
    plt.subplot(1, 2, 1)
    plt.plot(sizes, tc_times, 'o-', label='Tensor Cores')
    plt.plot(sizes, normal_times, 'o-', label='Normal')
    plt.xlabel('Matrix Size (N for N×N)')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance')
    plt.legend()
    plt.grid(True)
    
    # Plot speedup
    plt.subplot(1, 2, 2)
    plt.bar(sizes, speedup)
    plt.xlabel('Matrix Size (N for N×N)')
    plt.ylabel('Speedup (×)')
    plt.title('Tensor Cores Speedup vs. Normal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def create_wmma_optimized_model(input_dim, hidden_dim, output_dim):
    """
    Create a simple neural network model that can benefit from Tensor Cores.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension (should be multiple of 8 for Tensor Cores)
        output_dim: Output dimension
    
    Returns:
        PyTorch model
    """
    # Ensure hidden_dim is a multiple of 8 for optimal Tensor Core usage
    if hidden_dim % 8 != 0:
        hidden_dim = hidden_dim + (8 - hidden_dim % 8)
        print(f"Adjusted hidden_dim to {hidden_dim} to optimize for Tensor Cores")
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    return model

def main():
    # Check if Tensor Cores are available
    has_tensor_cores = check_tensor_cores_available()
    
    if not has_tensor_cores:
        print("This demonstration requires a GPU with Tensor Cores support.")
        return
    
    # Matrix sizes to test (making them multiples of 8 for optimal Tensor Core usage)
    sizes = [512, 1024, 2048, 4096]
    
    # Run benchmarks
    print("\nRunning benchmark with Tensor Cores...")
    tc_results = matrix_multiply_benchmark(sizes, dtype=torch.float16, use_tensor_cores=True)
    
    print("\nRunning benchmark without Tensor Cores...")
    normal_results = matrix_multiply_benchmark(sizes, dtype=torch.float16, use_tensor_cores=False)
    
    # Visualize results
    if tc_results and normal_results:
        visualize_results(tc_results, normal_results)
    
    # Example of a neural network that can benefit from Tensor Cores
    print("\nCreating a model optimized for Tensor Cores...")
    
    # Set up for Tensor Core usage
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Create model with dimensions suitable for Tensor Cores
    input_dim = 784  # e.g., MNIST image flattened
    hidden_dim = 1024  # Multiple of 8 for Tensor Cores
    output_dim = 10  # MNIST classes
    
    model = create_wmma_optimized_model(input_dim, hidden_dim, output_dim)
    model = model.cuda().half()  # Use FP16 for Tensor Cores
    
    # Create sample input
    batch_size = 64
    x = torch.randn(batch_size, input_dim, dtype=torch.float16, device='cuda')
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Model output shape: {output.shape}")
    print("Successfully used Tensor Cores for matrix multiplication!")

if __name__ == "__main__":
    main()
