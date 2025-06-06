#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Constants
#define NUM_AGENTS 4
#define OBS_DIM 10
#define ACTION_DIM 5
#define HIDDEN_DIM 64
#define BATCH_SIZE 32
#define STATE_DIM 20

// CUDA kernel for Mixing Network
// Computes Q_tot as a monotonic function of individual Q_i values
__global__ void mixing_network_kernel(
    float* agent_qs,      // Input: [batch_size, num_agents, action_dim]
    float* state,         // Input: [batch_size, state_dim]
    float* weights1,      // Hypernetwork weights for first layer
    float* biases1,       // Hypernetwork biases for first layer
    float* weights2,      // Hypernetwork weights for second layer
    float* biases2,       // Hypernetwork biases for second layer
    float* q_tot,         // Output: [batch_size, action_dim]
    int batch_size,
    int num_agents,
    int action_dim,
    int hidden_dim,
    int state_dim
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Temporary storage for hidden layer
    float hidden[HIDDEN_DIM];
    for (int i = 0; i < hidden_dim; i++) {
        hidden[i] = 0.0f;
    }

    // First layer: Generate weights from state (hypernetwork)
    for (int i = 0; i < hidden_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            hidden[i] += state[batch_idx * state_dim + j] * weights1[j * hidden_dim + i];
        }
        hidden[i] += biases1[i];
        hidden[i] = fmaxf(hidden[i], 0.0f); // ReLU for non-negativity (monotonicity)
    }

    // Second layer: Compute Q_tot from agent Q-values
    for (int a = 0; a < action_dim; a++) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float w = 0.0f;
            for (int j = 0; j < state_dim; j++) {
                w += state[batch_idx * state_dim + j] * weights2[j * hidden_dim + i];
            }
            w = fabsf(w); // Ensure non-negative weights for monotonicity
            for (int n = 0; n < num_agents; n++) {
                sum += w * agent_qs[batch_idx * num_agents * action_dim + n * action_dim + a];
            }
        }
        q_tot[batch_idx * action_dim + a] = sum + biases2[a];
    }
}

// Simple DRQN Agent Network (simplified as feedforward for demo)
__global__ void agent_network_kernel(
    float* obs,           // Input: [batch_size, num_agents, obs_dim]
    float* prev_actions,  // Input: [batch_size, num_agents, action_dim]
    float* weights,       // Agent network weights
    float* biases,        // Agent network biases
    float* q_values,      // Output: [batch_size, num_agents, action_dim]
    int batch_size,
    int num_agents,
    int obs_dim,
    int action_dim,
    int hidden_dim
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int agent_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size || agent_idx >= num_agents) return;

    float hidden[HIDDEN_DIM];
    for (int i = 0; i < hidden_dim; i++) {
        hidden[i] = 0.0f;
        for (int j = 0; j < obs_dim; j++) {
            hidden[i] += obs[batch_idx * num_agents * obs_dim + agent_idx * obs_dim + j] * 
                         weights[j * hidden_dim + i];
        }
        for (int j = 0; j < action_dim; j++) {
            hidden[i] += prev_actions[batch_idx * num_agents * action_dim + agent_idx * action_dim + j] * 
                         weights[(obs_dim + j) * hidden_dim + i];
        }
        hidden[i] += biases[i];
        hidden[i] = fmaxf(hidden[i], 0.0f); // ReLU
    }

    for (int a = 0; a < action_dim; a++) {
        float q = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            q += hidden[i] * weights[(obs_dim + action_dim) * hidden_dim + i * action_dim + a];
        }
        q += biases[hidden_dim + a];
        q_values[batch_idx * num_agents * action_dim + agent_idx * action_dim + a] = q;
    }
}

// Host function to orchestrate QMIX computation
void run_qmix(
    float* d_obs,           // Device: Observations
    float* d_prev_actions,  // Device: Previous actions
    float* d_state,         // Device: Global state
    float* d_agent_weights, // Device: Agent network weights
    float* d_agent_biases,  // Device: Agent network biases
    float* d_mix_weights1,  // Device: Mixing network weights (first layer)
    float* d_mix_biases1,   // Device: Mixing network biases (first layer)
    float* d_mix_weights2,  // Device: Mixing network weights (second layer)
    float* d_mix_biases2,   // Device: Mixing network biases (second layer)
    float* d_q_tot,         // Device: Output joint Q-values
    int batch_size,
    int num_agents,
    int obs_dim,
    int action_dim,
    int hidden_dim,
    int state_dim
) {
    dim3 blockDim(16, 16);
    dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, 
                 (num_agents + blockDim.y - 1) / blockDim.y);

    // Compute individual Q-values
    float* d_agent_qs;
    CUDA_CHECK(cudaMalloc(&d_agent_qs, batch_size * num_agents * action_dim * sizeof(float)));
    
    agent_network_kernel<<<gridDim, blockDim>>>(
        d_obs, d_prev_actions, d_agent_weights, d_agent_biases, d_agent_qs,
        batch_size, num_agents, obs_dim, action_dim, hidden_dim
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute joint Q-value via mixing network
    mixing_network_kernel<<<(batch_size + 15) / 16, 16>>>(
        d_agent_qs, d_state, d_mix_weights1, d_mix_biases1, d_mix_weights2, d_mix_biases2, d_q_tot,
        batch_size, num_agents, action_dim, hidden_dim, state_dim
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_agent_qs));
}

// Main function for testing
int main() {
    // Allocate and initialize dummy data
    float *h_obs, *h_prev_actions, *h_state, *h_agent_weights, *h_agent_biases;
    float *h_mix_weights1, *h_mix_biases1, *h_mix_weights2, *h_mix_biases2, *h_q_tot;
    float *d_obs, *d_prev_actions, *d_state, *d_agent_weights, *d_agent_biases;
    float *d_mix_weights1, *d_mix_biases1, *d_mix_weights2, *d_mix_biases2, *d_q_tot;

    // Allocate host memory
    h_obs = (float*)malloc(BATCH_SIZE * NUM_AGENTS * OBS_DIM * sizeof(float));
    h_prev_actions = (float*)malloc(BATCH_SIZE * NUM_AGENTS * ACTION_DIM * sizeof(float));
    h_state = (float*)malloc(BATCH_SIZE * STATE_DIM * sizeof(float));
    h_agent_weights = (float*)malloc((OBS_DIM + ACTION_DIM) * HIDDEN_DIM + HIDDEN_DIM * ACTION_DIM * sizeof(float));
    h_agent_biases = (float*)malloc((HIDDEN_DIM + ACTION_DIM) * sizeof(float));
    h_mix_weights1 = (float*)malloc(STATE_DIM * HIDDEN_DIM * sizeof(float));
    h_mix_biases1 = (float*)malloc(HIDDEN_DIM * sizeof(float));
    h_mix_weights2 = (float*)malloc(STATE_DIM * HIDDEN_DIM * sizeof(float));
    h_mix_biases2 = (float*)malloc(ACTION_DIM * sizeof(float));
    h_q_tot = (float*)malloc(BATCH_SIZE * ACTION_DIM * sizeof(float));

    // Initialize with dummy values (e.g., 1.0 for simplicity)
    for (int i = 0; i < BATCH_SIZE * NUM_AGENTS * OBS_DIM; i++) h_obs[i] = 1.0f;
    for (int i = 0; i < BATCH_SIZE * NUM_AGENTS * ACTION_DIM; i++) h_prev_actions[i] = 0.0f;
    for (int i = 0; i < BATCH_SIZE * STATE_DIM; i++) h_state[i] = 1.0f;
    for (int i = 0; i < (OBS_DIM + ACTION_DIM) * HIDDEN_DIM + HIDDEN_DIM * ACTION_DIM; i++) h_agent_weights[i] = 0.1f;
    for (int i = 0; i < HIDDEN_DIM + ACTION_DIM; i++) h_agent_biases[i] = 0.0f;
    for (int i = 0; i < STATE_DIM * HIDDEN_DIM; i++) h_mix_weights1[i] = 0.1f;
    for (int i = 0; i < HIDDEN_DIM; i++) h_mix_biases1[i] = 0.0f;
    for (int i = 0; i < STATE_DIM * HIDDEN_DIM; i++) h_mix_weights2[i] = 0.1f;
    for (int i = 0; i < ACTION_DIM; i++) h_mix_biases2[i] = 0.0f;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_obs, BATCH_SIZE * NUM_AGENTS * OBS_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev_actions, BATCH_SIZE * NUM_AGENTS * ACTION_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state, BATCH_SIZE * STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_agent_weights, ((OBS_DIM + ACTION_DIM) * HIDDEN_DIM + HIDDEN_DIM * ACTION_DIM) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_agent_biases, (HIDDEN_DIM + ACTION_DIM) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mix_weights1, STATE_DIM * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mix_biases1, HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mix_weights2, STATE_DIM * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mix_biases2, ACTION_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_tot, BATCH_SIZE * ACTION_DIM * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_obs, h_obs, BATCH_SIZE * NUM_AGENTS * OBS_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_prev_actions, h_prev_actions, BATCH_SIZE * NUM_AGENTS * ACTION_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state, h_state, BATCH_SIZE * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_agent_weights, h_agent_weights, ((OBS_DIM + ACTION_DIM) * HIDDEN_DIM + HIDDEN_DIM * ACTION_DIM) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_agent_biases, h_agent_biases, (HIDDEN_DIM + ACTION_DIM) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mix_weights1, h_mix_weights1, STATE_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mix_biases1, h_mix_biases1, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mix_weights2, h_mix_weights2, STATE_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mix_biases2, h_mix_biases2, ACTION_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // Run QMIX computation
    run_qmix(
        d_obs, d_prev_actions, d_state, d_agent_weights, d_agent_biases,
        d_mix_weights1, d_mix_biases1, d_mix_weights2, d_mix_biases2, d_q_tot,
        BATCH_SIZE, NUM_AGENTS, OBS_DIM, ACTION_DIM, HIDDEN_DIM, STATE_DIM
    );

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_q_tot, d_q_tot, BATCH_SIZE * ACTION_DIM * sizeof(float), cudaMemcpyDeviceToHost));

    // Print sample output
    std::cout << "Sample Q_tot values for first batch:\n";
    for (int a = 0; a < ACTION_DIM; a++) {
        std::cout << h_q_tot[a] << " ";
    }
    std::cout << "\n";

    // Free memory
    free(h_obs); free(h_prev_actions); free(h_state);
    free(h_agent_weights); free(h_agent_biases);
    free(h_mix_weights1); free(h_mix_biases1);
    free(h_mix_weights2); free(h_mix_biases2); free(h_q_tot);
    CUDA_CHECK(cudaFree(d_obs)); CUDA_CHECK(cudaFree(d_prev_actions));
    CUDA_CHECK(cudaFree(d_state)); CUDA_CHECK(cudaFree(d_agent_weights));
    CUDA_CHECK(cudaFree(d_agent_biases)); CUDA_CHECK(cudaFree(d_mix_weights1));
    CUDA_CHECK(cudaFree(d_mix_biases1)); CUDA_CHECK(cudaFree(d_mix_weights2));
    CUDA_CHECK(cudaFree(d_mix_biases2)); CUDA_CHECK(cudaFree(d_q_tot));

    return 0;
}