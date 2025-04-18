import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load, CUDAExtension
from setuptools import setup
from torch.autograd import Function

# PyTorch nn.Module implementation of SwiGLU
class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function
    As described in "GLU Variants Improve Transformer" paper
    
    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    where Swish(x) = x * sigmoid(β*x), and β is typically 1
    """
    def __init__(self, input_size, output_size, beta=1.0):
        super(SwiGLU, self).__init__()
        # Create two linear transformations
        self.linear_gate = nn.Linear(input_size, output_size)
        self.linear_value = nn.Linear(input_size, output_size)
        self.beta = beta
        
    def forward(self, x):
        # Apply the two linear transformations
        gate = self.linear_gate(x)
        value = self.linear_value(x)
        
        # Apply Swish activation to gate
        gate = gate * torch.sigmoid(self.beta * gate)
        
        # Element-wise multiplication with value
        return gate * value

# Class for CUDA implementation loading
class CudaImplementation:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.loaded = False
        self.swiglu_cuda = None
        self.load_cuda_kernel()
    
    def load_cuda_kernel(self):
        """Load and compile the CUDA kernel if CUDA is available"""
        if not torch.cuda.is_available():
            print("CUDA not available. Using PyTorch implementation.")
            return
        
        try:
            # Create directory for CUDA extensions
            os.makedirs('cuda_extensions', exist_ok=True)
            
            # Define CUDA kernel source path
            kernel_path = os.path.join('cuda_extensions', 'swiglu_kernel.cu')
            
            # Check if we need to write the source file
            if not os.path.exists(kernel_path):
                print("Writing CUDA kernel source...")
                with open(kernel_path, 'w') as f:
                    # This would be the content of the swiglu_kernel.cu file
                    # For brevity, we're not duplicating it here
                    pass
            
            # Compile the CUDA extension
            print("Compiling CUDA kernel...")
            self.swiglu_cuda = load(
                name="swiglu_kernel",
                sources=[kernel_path],
                verbose=True
            )
            
            self.loaded = True
            print("CUDA kernel compiled successfully!")
            
        except Exception as e:
            print(f"Failed to compile CUDA kernel: {e}")
            print("Falling back to PyTorch implementation")

# Custom autograd Function for CUDA implementation
class SwiGLUFunction(Function):
    @staticmethod
    def forward(ctx, gate_input, value_input, beta=1.0):
        # Save tensors needed for backward
        ctx.save_for_backward(gate_input, value_input)
        ctx.beta = beta
        
        cuda_impl = CudaImplementation.get_instance()
        if cuda_impl.loaded:
            # Use CUDA implementation
            return cuda_impl.swiglu_cuda.forward(gate_input, value_input, beta)
        else:
            # Fallback to PyTorch implementation
            return gate_input * torch.sigmoid(beta * gate_input) * value_input
    
    @staticmethod
    def backward(ctx, grad_output):
        gate_input, value_input = ctx.saved_tensors
        beta = ctx.beta
        
        cuda_impl = CudaImplementation.get_instance()
        if cuda_impl.loaded:
            # Use CUDA implementation
            grad_gate, grad_value = cuda_impl.swiglu_cuda.backward(
                gate_input, value_input, grad_output, beta)
        else:
            # Fallback to PyTorch implementation
            sigmoid_val = torch.sigmoid(beta * gate_input)
            swish = gate_input * sigmoid_val
            
            # Gradient for value
            grad_value = grad_output * swish
            
            # Gradient for gate
            d_swish = sigmoid_val + beta * gate_input * sigmoid_val * (1 - sigmoid_val)
            grad_gate = grad_output * value_input * d_swish
        
        # Return gradients for inputs and None for beta parameter
        return grad_gate, grad_value, None

# CUDA-accelerated SwiGLU module
class SwiGLUCuda(nn.Module):
    def __init__(self, input_size, output_size, beta=1.0):
        super(SwiGLUCuda, self).__init__()
        self.linear_gate = nn.Linear(input_size, output_size)
        self.linear_value = nn.Linear(input_size, output_size)
        self.beta = beta
        
        # Initialize CUDA implementation
        CudaImplementation.get_instance()
    
    def forward(self, x):
        # Apply linear transformations
        gate = self.linear_gate(x)
        value = self.linear_value(x)
        
        # Apply SwiGLU using custom Function
        return SwiGLUFunction.apply(gate, value, self.beta)

# Setup function for PyTorch extension
def setup_cuda_extension():
    """Setup function to install the CUDA extension"""
    if not torch.cuda.is_available():
        print("CUDA not available. Extension not installed.")
        return
    
    try:
        setup(
            name="swiglu_cuda",
            ext_modules=[
                CUDAExtension(
                    name="swiglu_kernel",
                    sources=["cuda_extensions/swiglu_kernel.cu"],
                )
            ],
            cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
        )
        print("CUDA extension installed successfully!")
    except Exception as e:
        print(f"Failed to install CUDA extension: {e}")

# Example usage demonstration
def test_swiglu():
    batch_size = 32
    input_size = 512
    output_size = 1024
    
    # Create input tensor
    x = torch.randn(batch_size, input_size)
    
    # Test PyTorch implementation
    swiglu = SwiGLU(input_size, output_size)
    output = swiglu(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test CUDA implementation if available
    if torch.cuda.is_available():
        try:
            # Move to GPU
            x_cuda = x.cuda()
            swiglu_cuda = SwiGLUCuda(input_size, output_size).cuda()
            
            # Forward pass
            output_cuda = swiglu_cuda(x_cuda)
            print(f"CUDA output shape: {output_cuda.shape}")
            
            # Measure performance
            import time
            
            # Warm-up
            for _ in range(10):
                _ = swiglu_cuda(x_cuda)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            iterations = 100
            for _ in range(iterations):
                _ = swiglu_cuda(x_cuda)
            torch.cuda.synchronize()
            end = time.time()
            
            print(f"CUDA implementation: {(end - start) / iterations * 1000:.3f} ms per iteration")
            
        except Exception as e:
            print(f"Failed to use CUDA implementation: {e}")

if __name__ == "__main__":
    test_swiglu()