import numpy as np
import ctypes
from ctypes import *
import os
import time

# Load the CUDA library
def load_cuda_lib():
    # Compile the CUDA code
    os.system("nvcc -Xcompiler -fPIC -shared -o libwarp_matmul.so warp_matmul.cu")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL('./libwarp_matmul.so')
    
    # Set the argument types
    cuda_lib.cuda_matmul.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
        c_int, c_int, c_int
    ]
    
    return cuda_lib

def warp_matmul(A, B):
    """
    Perform matrix multiplication using warp-level tiling
    
    Args:
        A: Numpy matrix of shape (M, K)
        B: Numpy matrix of shape (K, N)
    
    Returns:
        C: Result matrix of shape (M, N)
    """
    # Convert to float32 if needed
    if A.dtype != np.float32:
        A = A.astype(np.float32)
    if B.dtype != np.float32:
        B = B.astype(np.float32)
    
    # Get matrix dimensions
    M, K = A.shape
    K2, N = B.shape
    
    if K != K2:
        raise ValueError(f"Matrix dimensions don't match: A is {A.shape} and B is {B.shape}")
    
    # Create output matrix
    C = np.zeros((M, N), dtype=np.float32)
    
    # Get pointers to the numpy arrays
    A_ptr = A.ctypes.data_as(POINTER(c_float))
    B_ptr = B.ctypes.data_as(POINTER(c_float))
    C_ptr = C.ctypes.data_as(POINTER(c_float))
    
    # Load CUDA library
    cuda_lib = load_cuda_lib()
    
    # Call the CUDA function
    cuda_lib.cuda_matmul(A_ptr, B_ptr, C_ptr, M, N, K)
    
    return C

if __name__ == "__main__":
    # Test the matrix multiplication
    M, K, N = 1024, 1024, 1024
    
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    
    # Measure CUDA performance
    start = time.time()
    C_cuda = warp_matmul(A, B)
    cuda_time = time.time() - start
    
    # Measure NumPy performance
    start = time.time()
    C_numpy = np.matmul(A, B)
    numpy_time = time.time() - start
    
    # Verify results
    max_diff = np.max(np.abs(C_cuda - C_numpy))
    
    print(f"Matrix multiplication shape: ({M}, {K}) x ({K}, {N})")
    print(f"Maximum difference: {max_diff}")
    print(f"CUDA time: {cuda_time:.4f} seconds")
    print(f"NumPy time: {numpy_time:.4f} seconds")
    print(f"Speedup: {numpy_time/cuda_time:.2f}x")

