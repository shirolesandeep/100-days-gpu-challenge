import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Constants
STATE_SIZE = 4          # 4D state vector (x, y, goal_x, goal_y)
ACTION_SIZE = 2         # 2 actions (right, up)
HIDDEN_SIZE = 16        # Hidden layer size
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 0.001   # Learning rate
BATCH_SIZE = 32         # Batch size for replay
MEMORY_SIZE = 10000     # Replay memory size
EPSILON_START = 1.0     # Initial exploration rate
EPSILON_END = 0.1       # Final exploration rate
EPSILON_DECAY = 1000    # Decay steps
MAX_EPISODES = 1000     # Number of episodes
GRID_SIZE = 4           # 4x4 grid world

# Q-Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Grid world environment
class Environment:
    def __init__(self):
        self.state = [0, 0]  # (x, y) position
        self.goal = [GRID_SIZE - 1, GRID_SIZE - 1]  # Goal position
        self.is_done = False

    def reset(self):
        self.state = [0, 0]
        self.is_done = False
        return self.get_state()

    def get_state(self):
        return np.array([
            self.state[0] / GRID_SIZE,
            self.state[1] / GRID_SIZE,
            self.goal[0] / GRID_SIZE,
            self.goal[1] / GRID_SIZE
        ], dtype=np.float32)

    def step(self, action):
        next_state = self.state.copy()
        if action == 0:  # Right
            next_state[0] = min(next_state[0] + 1, GRID_SIZE - 1)
        else:  # Up
            next_state[1] = min(next_state[1] + 1, GRID_SIZE - 1)

        self.state = next_state
        reward = -0.01
        self.is_done = (self.state == self.goal)

        if self.is_done:
            reward = 1.0

        return self.get_state(), reward, self.is_done

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Training function
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Q-Network, optimizer, and memory
    q_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    env = Environment()
    epsilon = EPSILON_START
    step_count = 0

    for episode in range(MAX_EPISODES):
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        total_reward = 0.0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(ACTION_SIZE)
            else:
                with torch.no_grad():
                    q_values = q_network(state.unsqueeze(0))
                    action = q_values.argmax().item()

            # Take action
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            total_reward += reward

            # Store transition
            memory.push((state, action, reward, next_state, done))

            # Update state
            state = next_state
            step_count += 1

            # Update epsilon
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                      np.exp(-1. * step_count / EPSILON_DECAY)

            # Train if enough memory
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = tuple(zip(*transitions))

                states = torch.stack(batch[0]).to(device)
                actions = torch.LongTensor(batch[1]).to(device)
                rewards = torch.FloatTensor(batch[2]).to(device)
                next_states = torch.stack(batch[3]).to(device)
                dones = torch.FloatTensor(batch[4]).to(device)

                # Compute Q-values
                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = q_network(next_states).max(1)[0]
                    targets = rewards + (1 - dones) * GAMMA * next_q_values

                # Compute loss
                loss = nn.MSELoss()(q_values, targets)

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    train()