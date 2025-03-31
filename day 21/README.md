Day 21: This is a tiled matrix multiplication implementation inspired by FlashAttention's approach to reduce memory overhead. The CUDA version uses shared memory to load tiles of Q and K, performing the multiplication in chunks. The PyTorch version mimics this tiling strategy but relies on PyTorch's native matrix multiplication for simplicity.
Input: Q (query) and K (key) matrices of shape [N, d].
Output: S (attention scores) of shape [N, N].
Tile Size: Set to 16 for both implementations, but adjustable.
This Day 21 implementation provides the groundwork for the FlashAttention forward pass (Day 22), where softmax and scaling are added, and it evolves further in subsequent days with backward passes, optimizations, and RoPE.