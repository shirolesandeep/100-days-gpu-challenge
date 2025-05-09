import torch
import torch.multiprocessing as mp
import numpy as np
import time
from typing import Tuple

# Configuration
STATE_DIM = 4
ACTION_DIM = 2
NUM_EPISODES = 1000
MAX_STEPS = 200
NUM_SIMS = 1024  # Number of parallel simulations per GPU
NUM_GPUS = torch.cuda.device_count()
SIMS_PER_GPU = NUM_SIMS // NUM_GPUS

# Simple CartPole-like environment
class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.dt = 0.02

    def step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate one step of the environment."""
        x, x_dot, theta, theta_dot = state.chunk(4, dim=-1)
        
        force = torch.where(action == 1, 1.0, -1.0)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (force + self.masspole * self.length * theta_dot**2 * sintheta) / (self.masscart + self.masspole)
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                  (self.length * (4.0/3.0 - self.masspole * costheta**2 / (self.masscart + self.masspole)))
        xacc = temp - self.masspole * self.length * thetaacc * costheta / (self.masscart + self.masspole)

        x_dot = x_dot + xacc * self.dt
        x = x + x_dot * self.dt
        theta_dot = theta_dot + thetaacc * self.dt
        theta = theta + theta_dot * self.dt

        next_state = torch.cat([x, x_dot, theta, theta_dot], dim=-1)
        reward = torch.ones_like(x)
        done = (torch.abs(x) > 2.4) | (torch.abs(theta) > 12.0 * np.pi / 180.0)
        
        return next_state, reward, done

# Random policy
def select_action(state: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Select random actions."""
    return torch.randint(0, ACTION_DIM, size=(state.shape[0],), device=device)

# Simulation worker for each GPU
def run_simulations(gpu_id: int, episodes: int, results_queue: mp.Queue):
    """Run simulations on a single GPU."""
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    env = CartPoleEnv()
    
    # Initialize states
    states = torch.zeros((SIMS_PER_GPU, STATE_DIM), device=device)
    
    for episode in range(episodes):
        episode_rewards = torch.zeros(SIMS_PER_GPU, device=device)
        episode_lengths = torch.zeros(SIMS_PER_GPU, dtype=torch.int32, device=device)
        
        for step in range(MAX_STEPS):
            actions = select_action(states, device)
            next_states, rewards, dones = env.step(states, actions)
            
            episode_rewards += rewards * (1 - dones.float())
            episode_lengths += (1 - dones).int()
            
            states = torch.where(dones.unsqueeze(-1), torch.zeros_like(states), next_states)
            
            if torch.all(dones):
                break
        
        # Collect results
        avg_reward = episode_rewards.mean().item()
        avg_length = episode_lengths.float().mean().item()
        
        if episode % 100 == 0:
            results_queue.put((gpu_id, episode, avg_reward, avg_length))
    
    results_queue.put(None)  # Signal completion

def main():
    mp.set_start_method('spawn')
    results_queue = mp.Queue()
    processes = []
    
    # Start one process per GPU
    for gpu_id in range(NUM_GPUS):
        p = mp.Process(target=run_simulations, args=(gpu_id, NUM_EPISODES // NUM_GPUS, results_queue))
        p.start()
        processes.append(p)
    
    # Collect and print results
    completed = 0
    while completed < NUM_GPUS:
        result = results_queue.get()
        if result is None:
            completed += 1
        else:
            gpu_id, episode, avg_reward, avg_length = result
            print(f"GPU {gpu_id}, Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.2f}")
    
    # Cleanup
    for p in processes:
        p.join()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")