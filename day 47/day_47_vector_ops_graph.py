import torch
import time

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this script on a CUDA-enabled GPU.")
        return

    # Parameters
    N = 1 << 20  # 1M elements
    num_iterations = 100
    scalar = 2.0
    device = torch.device("cuda")

    # Initialize tensors
    a = torch.rand(N, device=device)
    b = torch.rand(N, device=device)
    c = torch.zeros(N, device=device)
    temp = torch.zeros(N, device=device)

    # Create a stream for graph capture
    stream = torch.cuda.Stream()

    # Warm-up run (to avoid initial overhead)
    with torch.cuda.stream(stream):
        temp.copy_(a + b)
        c.copy_(temp * a)
        c.mul_(scalar)
    torch.cuda.synchronize()

    # Capture the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        temp.copy_(a + b)
        c.copy_(temp * a)
        c.mul_(scalar)

    # Measure execution time with CUDA Graph
    start = time.time()
    for _ in range(num_iterations):
        graph.replay()
    torch.cuda.synchronize()
    graph_time = (time.time() - start) * 1000  # Convert to ms
    print(f"CUDA Graph execution time: {graph_time:.2f} ms")

    # Store graph result for verification
    c_graph = c.clone()

    # Reset output tensor
    c.zero_()

    # Measure execution time without CUDA Graph
    start = time.time()
    for _ in range(num_iterations):
        with torch.cuda.stream(stream):
            temp.copy_(a + b)
            c.copy_(temp * a)
            c.mul_(scalar)
    torch.cuda.synchronize()
    no_graph_time = (time.time() - start) * 1000  # Convert to ms
    print(f"Non-Graph execution time: {no_graph_time:.2f} ms")

    # Verify results
    max_diff = torch.max(torch.abs(c - c_graph)).item()
    if max_diff < 1e-5:
        print("Verification passed: Graph and non-graph results match")
    else:
        print(f"Verification failed: Max difference = {max_diff}")

if __name__ == "__main__":
    main()