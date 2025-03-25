Day 15: Sparse Matrix-Vector Multiplication (SpMV) with CSR Format
Concept: Sparse matrices (where most elements are zero) are common in many applications (e.g., graph algorithms, machine learning). Day 15 introduces sparse matrix-vector multiplication (SpMV) using the Compressed Sparse Row (CSR) format, which is memory-efficient for sparse data. This builds on matrix operations and introduces a new challenge of handling irregular data access patterns.
CUDA C Code (Day 15 - SpMV with CSR)
This implementation performs SpMV using the CSR format, where the sparse matrix is stored as three arrays: values (non-zero elements), column indices, and row pointers.
