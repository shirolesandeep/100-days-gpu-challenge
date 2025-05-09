#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple environment: CartPole-like simulation
#define STATE_DIM 4
#define ACTION_DIM 2
#define NUM_EPISODES 1000
#define MAX_STEPS 200
#define NUM_SIMS 1024 // Number of parallel simulations

// Kernel to initialize random states
__global__ void init_random_states(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_SIMS) {
        curand_init(seed, idx, 0, states + idx);
    }
}

// Simple policy: random action selection
__device__ int select_action(curandState* state) {
    float rand_val = curand_uniform(state);
    return (rand_val < 0.5f) ? 0 : 1;
}

// Environment step function (simplified CartPole dynamics)
__device__ void env_step(float* state, int action, float* next_state, float* reward, int* done) {
    float x = state[0];
    float x_dot = state[1];
    float theta = state[2];
    float theta_dot = state[3];

    float force = (action == 1) ? 1.0f : -1.0f;
    float gravity = 9.8f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float length = 0.5f;
    float dt = 0.02f;

    float costheta = cosf(theta);
    float sintheta = sinf(theta);

    float temp = (force + masspole * length * theta_dot * theta_dot * sintheta) / (masscart + masspole);
    float thetaacc = (gravity * sintheta - costheta * temp) / 
                     (length * (4.0f/3.0f - masspole * costheta * costheta / (masscart + masspole)));
    float xacc = temp - masspole * length * thetaacc * costheta / (masscart + masspole);

    x_dot += xacc * dt;
    x += x_dot * dt;
    theta_dot += thetaacc * dt;
    theta += theta_dot * dt;

    next_state[0] = x;
    next_state[1] = x_dot;
    next_state[2] = theta;
    next_state[3] = theta_dot;

    *reward = 1.0f;
    *done = (fabsf(x) > 2.4f || fabsf(theta) > 12.0f * M_PI / 180.0f) ? 1 : 0;
}

// Kernel for parallel RL simulations
__global__ void run_simulations(curandState* rand_states, float* total_rewards, int* episode_lengths) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= NUM_SIMS) return;

    curandState local_state = rand_states[sim_idx];
    float state[STATE_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
    float next_state[STATE_DIM];
    float reward;
    int done;
    float episode_reward = 0.0f;
    int steps = 0;

    for (int step = 0; step < MAX_STEPS; step++) {
        int action = select_action(&local_state);
        env_step(state, action, next_state, &reward, &done);
        
        episode_reward += reward;
        steps++;
        
        for (int i = 0; i < STATE_DIM; i++) {
            state[i] = next_state[i];
        }

        if (done) break;
    }

    total_rewards[sim_idx] = episode_reward;
    episode_lengths[sim_idx] = steps;
    rand_states[sim_idx] = local_state;
}

int main() {
    // Allocate memory
    curandState* d_rand_states;
    float* d_total_rewards;
    int* d_episode_lengths;
    float* h_total_rewards = (float*)malloc(NUM_SIMS * sizeof(float));
    int* h_episode_lengths = (int*)malloc(NUM_SIMS * sizeof(int));

    cudaMalloc(&d_rand_states, NUM_SIMS * sizeof(curandState));
    cudaMalloc(&d_total_rewards, NUM_SIMS * sizeof(float));
    cudaMalloc(&d_episode_lengths, NUM_SIMS * sizeof(int));

    // Initialize random states
    dim3 block(256);
    dim3 grid((NUM_SIMS + block.x - 1) / block.x);
    init_random_states<<<grid, block>>>(d_rand_states, time(NULL));
    cudaDeviceSynchronize();

    // Run simulations across multiple GPUs
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        run_simulations<<<grid, block, 0, stream>>>(d_rand_states, d_total_rewards, d_episode_lengths);
        cudaStreamSynchronize(stream);

        // Copy results back
        cudaMemcpy(h_total_rewards, d_total_rewards, NUM_SIMS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_episode_lengths, d_episode_lengths, NUM_SIMS * sizeof(int), cudaMemcpyDeviceToHost);

        // Calculate average reward
        float avg_reward = 0.0f;
        float avg_length = 0.0f;
        for (int i = 0; i < NUM_SIMS; i++) {
            avg_reward += h_total_rewards[i];
            avg_length += h_episode_lengths[i];
        }
        avg_reward /= NUM_SIMS;
        avg_length /= NUM_SIMS;

        if (episode % 100 == 0) {
            printf("Episode %d: Avg Reward = %.2f, Avg Length = %.2f\n", 
                   episode, avg_reward, avg_length);
        }
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_rand_states);
    cudaFree(d_total_rewards);
    cudaFree(d_episode_lengths);
    free(h_total_rewards);
    free(h_episode_lengths);

    return 0;
}