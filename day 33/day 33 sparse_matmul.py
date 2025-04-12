import numpy as np
import ctypes
from ctypes import *
import os
import scipy.sparse as sp
import time

# Load the CUDA library
def load_cuda_lib():
    # Compile the CUDA code
    os.system("nvcc -Xcompiler -fPIC -shared -o libsparse_matmul.so sparse_matmul.cu")
    
    # Load the compiled library
    cuda_lib = ctypes.CDLL('./libsparse_matmul.so')
    
    # Set the argument types
    cuda_lib.cuda_sparse_matmul.argtypes = [
        POINTER(c_float),  # values
        POINTER(c_int),    # colIndices
        POINTER(c_int),    # rowPointers
        POINTER(c_float),  # denseMatrix
        POINTER(c_float),  # result
        c_int, c_int, c_int, c_int  # M, N, K, nnz
    ]
    
    return cuda_lib

def sparse_matmul(sparse_mat, dense_mat):
    """
    Perform multiplication of a sparse matrix with a dense matrix using warp-level optimizations
    
    Args:
        sparse_mat: Scipy CSR sparse matrix of shape (M, K)
        dense_mat: Numpy dense matrix of shape (K, N)
    
    Returns:
        result: Dense matrix of shape (M, N)
    """
    # Convert to CSR format if needed
    if not sp.isspmatrix_csr(sparse_mat):
        sparse_mat = sparse_mat.tocsr()
    
    # Convert dense matrix to float32 if needed
    if dense_mat.dtype != np.float32:
        dense_mat = dense_mat.astype(np.float32)
    
    # Get matrix dimensions
    M, K = sparse_mat.shape
    K2, N = dense_mat.shape
    
    if K != K2:
        raise ValueError(f"Matrix dimensions don't match: sparse is {sparse_mat.shape} and dense is {dense_mat.shape}")
    
    # Extract CSR components
    values = sparse_mat.data.astype(np.float32)
    col_indices = sparse_mat.indices.astype(np.int32)
    row_pointers = sparse_mat.indptr.astype(np.int32)
    nnz = len(values)
    
    # Create output matrix
    result = np.zeros((M, N), dtype=np.float32)
    
    # Get pointers to the numpy arrays
    values_ptr = values.ctypes.data_as(POINTER(c_float))
    col_indices_ptr = col_indices.ctypes.data_as(POINTER(c_int))
    row_pointers_ptr = row_pointers.ctypes.data_as(POINTER(c_int))
    dense_ptr = dense_mat.ctypes.data_as(POINTER(c_float))
    result_ptr = result.ctypes.data_as(POINTER(c_float))
    
    # Load CUDA library
    cuda_lib = load_cuda_lib()
    
    # Call the CUDA function
    cuda_lib.cuda_sparse_matmul(
        values_ptr, col_indices_ptr, row_pointers_ptr,
        dense_ptr, result_ptr,
        M, N, K, nnz
    )
    
    return result

if __name__ == "__main__":
    # Test the sparse matrix multiplication
    M, K, N = 1024, 1024, 1024
    sparsity = 0.99  # 99% zeros
    
    # Create a random sparse matrix
    sparse_matrix = sp.random(M, K, density=1-sparsity, format='csr', dtype=np.float32)
    dense_matrix = np.random.rand(K, N).astype(np.float32)
    
    # Measure CUDA performance
    start = time.time()
    cuda_result = sparse_matmul(sparse_matrix, dense_matrix)
    cuda_time = time.time() - start
    
    # Measure SciPy performance
    start = time.time()
    scipy_result = sparse_matrix @ dense_matrix
    scipy_time = time.time() - start
    
    # Verify results
    max_diff = np.max(np.abs(cuda_result - scipy_result))
    
    print(f"Sparse matrix multiplication shape: ({M}, {K}) x ({K}, {N})")
    print(f"Sparsity: {sparsity*100:.1f}%")
    print(f"Non-zero elements: {sparse_matrix.count_nonzero()}")
    print(f"Maximum difference: {max_diff}")
    print(f"CUDA time: {cuda_time:.4f} seconds")
    print(f"SciPy time: {scipy_time:.4f} seconds")
    print(f"Speedup: {scipy_time/cuda_time:.2f}x")
