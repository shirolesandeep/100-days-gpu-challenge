import numpy as np
import ctypes
from ctypes import *
import os

# Load the CUDA library
def load_cuda_lib():
    # Compile the CUDA code
    os.system("nvcc -Xcompiler -fPIC -shared -o libwarp_reduction.so warp_reduction.cu")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL('./libwarp_reduction.so')
    
    # Set the argument and return types
    cuda_lib.cuda_reduce.argtypes = [POINTER(c_float), c_int]
    cuda_lib.cuda_reduce.restype = c_float
    
    return cuda_lib

def warp_reduction(arr):
    """
    Perform reduction on array using warp-level shuffle intrinsics
    
    Args:
        arr: Numpy array of float32 values
    
    Returns:
        Sum of all elements in the array
    """
    # Convert to float32 if needed
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    
    # Get pointer to the numpy array
    arr_ptr = arr.ctypes.data_as(POINTER(c_float))
    
    # Load CUDA library
    cuda_lib = load_cuda_lib()
    
    # Call the CUDA function
    result = cuda_lib.cuda_reduce(arr_ptr, len(arr))
    
    return result

if __name__ == "__main__":
    # Test the reduction
    size = 1000000
    data = np.random.rand(size).astype(np.float32)
    
    # Verify results
    cuda_sum = warp_reduction(data)
    numpy_sum = np.sum(data)
    
    print(f"CUDA warp reduction sum: {cuda_sum}")
    print(f"NumPy sum: {numpy_sum}")
    print(f"Difference: {abs(cuda_sum - numpy_sum)}")



