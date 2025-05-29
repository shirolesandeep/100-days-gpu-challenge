import torch
import numpy as np
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 5x5 Grid World: 0=empty, 1=goal
grid = np.zeros((5, 5))
grid[4, 4] = 1  # Goal at (4,4)
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

# Initialize Q-table
q_table = torch.zeros((5, 5, 4), device=device)  # 5x5 grid, 4 actions
alpha, gamma, epsilon = 0.1, 0.9, 0.1
episodes = 1000

def get_next_state(state, action):
    next_state = (state[0] + action[0], state[1] + action[1])
    if 0 <= next_state[0] < 5 and 0 <= next_state[1] < 5:
        return next_state
    return state

def get_reward(state):
    return grid[state[0], state[1]]

# Training loop
for episode in range(episodes):
    state = (0, 0)  # Start at (0,0)
    steps = 0
    while grid[state[0], state[1]] != 1:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = torch.argmax(q_table[state[0], state[1]]).item()
        action = actions[action_idx]
        
        # Get next state and reward
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Q-learning update
        q_value = q_table[state[0], state[1], action_idx]
        next_max = torch.max(q_table[next_state[0], next_state[1]]).item()
        q_table[state[0], state[1], action_idx] = q_value + alpha * (reward + gamma * next_max - q_value)
        
        state = next_state
        steps += 1
    if episode % 100 == 0:
        print(f"Episode {episode}, Steps to goal: {steps}")

# Test the policy
state = (0, 0)
path = [state]
while grid[state[0], state[1]] != 1:
    action_idx = torch.argmax(q_table[state[0], state[1]]).item()
    state = get_next_state(state, actions[action_idx])
    path.append(state)
print("Learned path to goal:", path)
print("Q-table (final):", q_table.cpu().numpy())
