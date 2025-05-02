import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.LongTensor(action).unsqueeze(1).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# DQN implementation
class DQN:
    def __init__(self, state_dim, action_dim, buffer_capacity, gamma, lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_capacity, state_dim)
        self.steps = 0
        self.batch_size = 32
        self.target_update = 100

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

# Example usage
if __name__ == "__main__":
    state_dim = 4  # Example: CartPole-v1 state space
    action_dim = 2  # Example: CartPole-v1 action space
    buffer_capacity = 10000
    gamma = 0.99
    lr = 1e-3
    epsilon_start = 1.0
    epsilon_end = 0.02
    epsilon_decay = 0.995
    num_episodes = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(state_dim, action_dim, buffer_capacity, gamma, lr)
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = np.zeros(state_dim)  # Initialize state (replace with actual env)
        total_reward = 0.0

        while True:
            action = dqn.select_action(state, epsilon)
            # Simulate environment step (replace with actual environment)
            next_state = np.zeros(state_dim)
            reward = 0.0
            done = False

            dqn.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon}")
