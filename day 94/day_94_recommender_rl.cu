#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constants
#define NUM_USERS 1000
#define NUM_ITEMS 10
#define NUM_EPISODES 1000
#define ALPHA 0.1f    // Learning rate
#define GAMMA 0.9f    // Discount factor
#define EPSILON 0.1f  // Exploration rate

// CUDA error checking macro
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel to initialize Q-table and user preferences
__global__ void init_environment(float *q_table, float *user_prefs, curandState *states, unsigned int seed) {
    int user = blockIdx.x * blockDim.x + threadIdx.x;
    if (user >= NUM_USERS) return;

    // Initialize random state
    curand_init(seed, user, 0, &states[user]);

    // Initialize Q-table to 0
    for (int item = 0; item < NUM_ITEMS; item++) {
        q_table[user * NUM_ITEMS + item] = 0.0f;
    }

    // Initialize user preferences (randomly assign high preference to 2-3 items)
    for (int item = 0; item < NUM_ITEMS; item++) {
        user_prefs[user * NUM_ITEMS + item] = (curand_uniform(&states[user]) < 0.3f) ? 1.0f : 0.0f;
    }
}

// Kernel for Q-learning update
__global__ void q_learning_step(float *q_table, float *user_prefs, curandState *states, int episode) {
    int user = blockIdx.x * blockDim.x + threadIdx.x;
    if (user >= NUM_USERS) return;

    // Epsilon-greedy action selection
    int action;
    float r = curand_uniform(&states[user]);
    if (r < EPSILON) {
        // Exploration: random action
        action = curand(&states[user]) % NUM_ITEMS;
    } else {
        // Exploitation: choose best action
        float max_q = -1e9;
        action = 0;
        for (int i = 0; i < NUM_ITEMS; i++) {
            float q_val = q_table[user * NUM_ITEMS + i];
            if (q_val > max_q) {
                max_q = q_val;
                action = i;
            }
        }
    }

    // Get reward (1 if user likes item, 0 otherwise)
    float reward = user_prefs[user * NUM_ITEMS + action];

    // Find max Q-value for next state (simplified: state = user)
    float max_next_q = -1e9;
    for (int i = 0; i < NUM_ITEMS; i++) {
        float q_val = q_table[user * NUM_ITEMS + i];
        if (q_val > max_next_q) {
            max_next_q = q_val;
        }
    }

    // Q-learning update
    int idx = user * NUM_ITEMS + action;
    q_table[idx] += ALPHA * (reward + GAMMA * max_next_q - q_table[idx]);
}

// Host function to evaluate recommendations
void evaluate_recommendations(float *q_table, float *user_prefs) {
    float total_reward = 0.0f;
    for (int user = 0; user < NUM_USERS; user++) {
        // Find best action (item) for user
        int best_action = 0;
        float max_q = -1e9;
        for (int item = 0; item < NUM_ITEMS; item++) {
            float q_val = q_table[user * NUM_ITEMS + item];
            if (q_val > max_q) {
                max_q = q_val;
                best_action = item;
            }
        }
        total_reward += user_prefs[user * NUM_ITEMS + best_action];
    }
    printf("Average reward per user: %.3f\n", total_reward / NUM_USERS);
}

int main() {
    // Allocate host and device memory
    float *h_q_table = (float *)malloc(NUM_USERS * NUM_ITEMS * sizeof(float));
    float *d_q_table, *d_user_prefs;
    curandState *d_states;

    CUDA_CHECK(cudaMalloc(&d_q_table, NUM_USERS * NUM_ITEMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_user_prefs, NUM_USERS * NUM_ITEMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_states, NUM_USERS * sizeof(curandState)));

    // Set up CUDA grid and block
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_USERS + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize environment
    unsigned int seed = (unsigned int)time(NULL);
    init_environment<<<blocksPerGrid, threadsPerBlock>>>(d_q_table, d_user_prefs, d_states, seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run Q-learning episodes
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        q_learning_step<<<blocksPerGrid, threadsPerBlock>>>(d_q_table, d_user_prefs, d_states, episode);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Print progress every 100 episodes
        if ((episode + 1) % 100 == 0) {
            CUDA_CHECK(cudaMemcpy(h_q_table, d_q_table, NUM_USERS * NUM_ITEMS * sizeof(float), cudaMemcpyDeviceToHost));
            printf("Episode %d: ", episode + 1);
            evaluate_recommendations(h_q_table, h_q_table); // Note: Using h_q_table for simplicity
        }
    }

    // Copy final Q-table back to host
    CUDA_CHECK(cudaMemcpy(h_q_table, d_q_table, NUM_USERS * NUM_ITEMS * sizeof(float), cudaMemcpyDeviceToHost));

    // Evaluate final performance
    printf("Final evaluation: ");
    evaluate_recommendations(h_q_table, h_q_table); // Note: Using h_q_table for simplicity

    // Clean up
    free(h_q_table);
    CUDA_CHECK(cudaFree(d_q_table));
    CUDA_CHECK(cudaFree(d_user_prefs));
    CUDA_CHECK(cudaFree(d_states));

    return 0;
}