#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// CUDA kernels
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorMul(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void vectorScale(float *a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}

int main() {
    const int N = 1 << 20; // 1M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const float scalar = 2.0f;

    // Host arrays
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device arrays
    float *d_a, *d_b, *d_c, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA Graph objects
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Start capturing
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Launch kernels in sequence
    vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_temp, N);
    vectorMul<<<gridSize, blockSize, 0, stream>>>(d_temp, d_a, d_c, N);
    vectorScale<<<gridSize, blockSize, 0, stream>>>(d_c, scalar, N);

    // End capture and create graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Create executable graph
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Measure execution time with graphs
    auto start = std::chrono::high_resolution_clock::now();
    const int numIterations = 100;
    for (int i = 0; i < numIterations; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    auto graphTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("CUDA Graph execution time: %.2f ms\n", graphTime / 1000.0);

    // Measure execution time without graphs
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; i++) {
        vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_temp, N);
        vectorMul<<<gridSize, blockSize, 0, stream>>>(d_temp, d_a, d_c, N);
        vectorScale<<<gridSize, blockSize, 0, stream>>>(d_c, scalar, N);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    end = std::chrono::high_resolution_clock::now();
    auto noGraphTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    printf("Non-Graph execution time: %.2f ms\n", noGraphTime / 1000.0);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < 10; i++) { // Check first 10 elements
        float expected = (h_a[i] + h_b[i]) * h_a[i] * scalar;
        if (fabs(h_c[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: %f != %f\n", i, h_c[i], expected);
            break;
        }
    }
    printf("Verification completed\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}