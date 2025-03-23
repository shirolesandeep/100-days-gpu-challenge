Key Concept - Memory Coalescing:

In CUDA, threads in a warp (32 threads) should access contiguous memory locations to coalesce memory transactions. In Day 12, threads accessed global memory in a strided manner (e.g., A[row * n + a_col]), which can lead to uncoalesced accesses.
Day 13 ensures that consecutive threads in a warp load consecutive elements (e.g., threadIdx.x maps to consecutive columns in A and rows in B), improving memory throughput.
In PyTorch, we use .contiguous() to ensure that tensor memory is laid out contiguously, which helps the underlying cuBLAS library perform coalesced accesses.
