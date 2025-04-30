#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define constants
#define STATE_SIZE 4        // Example: 4D state vector (e.g., position, velocity)
#define ACTION_SIZE 2       // Example: 2 actions (e.g., left, right)
#define HIDDEN_SIZE 16      // Hidden layer size
#define BATCH_SIZE 32       // Batch size for training
#define GAMMA 0.99          // Discount factor
#define LEARNING_RATE 0.001 // Learning rate
#define MAX_EPISODES 1000   // Number of episodes
#define GRID_SIZE 4         // 4x4 grid world

// Neural network parameters
#define W1_SIZE (STATE_SIZE * HIDDEN_SIZE)  // Input to hidden weights
#define W2_SIZE (HIDDEN_SIZE * ACTION_SIZE) // Hidden to output weights

// CUDA error checking macro
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Matrix multiplication kernel
__global__ void matrixMul(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// Activation function (ReLU) kernel
__global__ void relu(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Gradient update kernel
__global__ void updateWeights(float *weights, float *gradients, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradients[idx];
    }
}

// Initialize random weights
__global__ void initWeights(curandState *state, float *weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState localState = state[idx];
        weights[idx] = curand_normal(&localState) * 0.01f;
        state[idx] = localState;
    }
}

// Simple grid world environment
struct Environment {
    int state[2]; // (x, y) position in 4x4 grid
    int goal[2];  // Goal position
    bool isDone;

    __host__ __device__ void reset() {
        state[0] = 0;
        state[1] = 0;
        goal[0] = GRID_SIZE - 1;
        goal[1] = GRID_SIZE - 1;
        isDone = false;
    }

    __host__ __device__ void getState(float *stateVec) {
        stateVec[0] = state[0] / (float)GRID_SIZE;
        stateVec[1] = state[1] / (float)GRID_SIZE;
        stateVec[2] = goal[0] / (float)GRID_SIZE;
        stateVec[3] = goal[1] / (float)GRID_SIZE;
    }

    __host__ __device__ float step(int action, int *nextState) {
        nextState[0] = state[0];
        nextState[1] = state[1];
        if (action == 0) nextState[0] = min(nextState[0] + 1, GRID_SIZE - 1); // Right
        else nextState[1] = min(nextState[1] + 1, GRID_SIZE - 1); // Up

        state[0] = nextState[0];
        state[1] = nextState[1];

        if (state[0] == goal[0] && state[1] == goal[1]) {
            isDone = true;
            return 1.0f;
        }
        return -0.01f;
    }
};

// Forward pass
void forward(float *d_input, float *d_W1, float *d_W2, float *d_hidden, float *d_output, dim3 blockDim, dim3 gridDim) {
    matrixMul<<<gridDim, blockDim>>>(d_input, d_W1, d_hidden, 1, STATE_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    relu<<<(HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    matrixMul<<<gridDim, blockDim>>>(d_hidden, d_W2, d_output, 1, HIDDEN_SIZE, ACTION_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Compute loss and gradients (simplified)
void computeGradients(float *d_output, float *d_target, float *d_W2_grad, float *d_hidden, float *d_W1_grad, float *d_input, dim3 blockDim, dim3 gridDim) {
    // Simplified: Compute output layer gradients
    // In practice, implement backpropagation properly
    matrixMul<<<gridDim, blockDim>>>(d_hidden, d_output, d_W2_grad, HIDDEN_SIZE, 1, ACTION_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    matrixMul<<<gridDim, blockDim>>>(d_input, d_hidden, d_W1_grad, STATE_SIZE, 1, HIDDEN_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    // Initialize environment
    Environment env;
    env.reset();

    // Allocate host and device memory
    float *h_input = (float *)malloc(STATE_SIZE * sizeof(float));
    float *h_output = (float *)malloc(ACTION_SIZE * sizeof(float));
    float *h_target = (float *)malloc(ACTION_SIZE * sizeof(float));
    float *d_input, *d_W1, *d_W2, *d_hidden, *d_output, *d_W1_grad, *d_W2_grad;
    curandState *d_state;

    CUDA_CHECK(cudaMalloc(&d_input, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, ACTION_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1_grad, W1_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2_grad, W2_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state, W1_SIZE * sizeof(curandState)));

    // Initialize weights
    dim3 blockDim(16, 16);
    dim3 gridDim((ACTION_SIZE + blockDim.x - 1) / blockDim.x, (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y);
    initWeights<<<(W1_SIZE + 255) / 256, 256>>>(d_state, d_W1, W1_SIZE);
    initWeights<<<(W2_SIZE + 255) / 256, 256>>>(d_state, d_W2, W2_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Training loop
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        env.reset();
        float totalReward = 0.0f;

        while (!env.isDone) {
            // Get state
            env.getState(h_input);
            CUDA_CHECK(cudaMemcpy(d_input, h_input, STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            // Forward pass
            forward(d_input, d_W1, d_W2, d_hidden, d_output, blockDim, gridDim);
            CUDA_CHECK(cudaMemcpy(h_output, d_output, ACTION_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            // Choose action (epsilon-greedy)
            int action = (rand() % 100 < 10) ? rand() % ACTION_SIZE : (h_output[0] > h_output[1] ? 0 : 1);

            // Take action
            int nextState[2];
            float reward = env.step(action, nextState);
            totalReward += reward;

            // Compute target Q-value (simplified)
            float maxNextQ = 0.0f; // In practice, compute via forward pass on next state
            h_target[action] = reward + (env.isDone ? 0.0f : GAMMA * maxNextQ);

            // Compute gradients and update weights
            CUDA_CHECK(cudaMemcpy(d_target, h_target, ACTION_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            computeGradients(d_output, d_target, d_W2_grad, d_hidden, d_W1_grad, d_input, blockDim, gridDim);
            updateWeights<<<(W1_SIZE + 255) / 256, 256>>>(d_W1, d_W1_grad, LEARNING_RATE, W1_SIZE);
            updateWeights<<<(W2_SIZE + 255) / 256, 256>>>(d_W2, d_W2_grad, LEARNING_RATE, W2_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (episode % 100 == 0) {
            printf("Episode %d, Total Reward: %.2f\n", episode, totalReward);
        }
    }

    // Cleanup
    free(h_input); free(h_output); free(h_target);
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_W1)); CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_W1_grad)); CUDA_CHECK(cudaFree(d_W2_grad)); CUDA_CHECK(cudaFree(d_state));

    return 0;
}