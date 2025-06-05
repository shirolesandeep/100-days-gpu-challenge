import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Constants
NUM_AGENTS = 2
STATE_DIM = 10
ACTION_DIM = 2
HIDDEN_DIM = 128
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 1000000
GAMMA = 0.99
TAU = 0.01
LR_ACTOR = 0.001
LR_CRITIC = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # Tanh to bound actions
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim * NUM_AGENTS + action_dim * NUM_AGENTS, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)  # Concatenate all states and actions
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = torch.FloatTensor(np.array(state)).to(DEVICE)
        action = torch.FloatTensor(np.array(action)).to(DEVICE)
        reward = torch.FloatTensor(np.array(reward)).to(DEVICE)
        next_state = torch.FloatTensor(np.array(next_state)).to(DEVICE)
        done = torch.FloatTensor(np.array(done)).to(DEVICE)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# MADDPG Agent
class MADDPG:
    def __init__(self):
        self.actors = [Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE) for _ in range(NUM_AGENTS)]
        self.critics = [Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE) for _ in range(NUM_AGENTS)]
        self.target_actors = [Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE) for _ in range(NUM_AGENTS)]
        self.target_critics = [Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE) for _ in range(NUM_AGENTS)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=LR_CRITIC) for critic in self.critics]
        self.buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.noise = [OUNoise(ACTION_DIM) for _ in range(NUM_AGENTS)]

        # Initialize target networks
        for actor, target_actor in zip(self.actors, self.target_actors):
            target_actor.load_state_dict(actor.state_dict())
        for critic, target_critic in zip(self.critics, self.target_critics):
            target_critic.load_state_dict(critic.state_dict())

    def act(self, states, add_noise=True):
        actions = []
        states = torch.FloatTensor(states).to(DEVICE)
        for i, actor in enumerate(self.actors):
            state = states[i].unsqueeze(0)
            action = actor(state).detach().cpu().numpy()[0]
            if add_noise:
                action += self.noise[i].sample()
            actions.append(np.clip(action, -1, 1))  # Assuming action bounds [-1, 1]
        return np.array(actions)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Reshape for processing
        states = states.view(BATCH_SIZE, NUM_AGENTS, STATE_DIM)
        actions = actions.view(BATCH_SIZE, NUM_AGENTS, ACTION_DIM)
        rewards = rewards.view(BATCH_SIZE, NUM_AGENTS)
        next_states = next_states.view(BATCH_SIZE, NUM_AGENTS, STATE_DIM)
        dones = dones.view(BATCH_SIZE, NUM_AGENTS)

        for i in range(NUM_AGENTS):
            # Critic update
            next_actions = torch.zeros(BATCH_SIZE, NUM_AGENTS, ACTION_DIM).to(DEVICE)
            for j in range(NUM_AGENTS):
                next_actions[:, j, :] = self.target_actors[j](next_states[:, j, :])
            next_actions = next_actions.view(BATCH_SIZE, NUM_AGENTS * ACTION_DIM)
            states_flat = states.view(BATCH_SIZE, NUM_AGENTS * STATE_DIM)
            next_states_flat = next_states.view(BATCH_SIZE, NUM_AGENTS * STATE_DIM)
            actions_flat = actions.view(BATCH_SIZE, NUM_AGENTS * ACTION_DIM)

            target_q = self.target_critics[i](next_states_flat, next_actions)
            target_q = rewards[:, i] + (1 - dones[:, i]) * GAMMA * target_q
            current_q = self.critics[i](states_flat, actions_flat)

            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Actor update
            pred_actions = torch.zeros(BATCH_SIZE, NUM_AGENTS, ACTION_DIM).to(DEVICE)
            for j in range(NUM_AGENTS):
                pred_actions[:, j, :] = self.actors[j](states[:, j, :])
            pred_actions_flat = pred_actions.view(BATCH_SIZE, NUM_AGENTS * ACTION_DIM)
            actor_loss = -self.critics[i](states_flat, pred_actions_flat).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # Soft update target networks
            self.soft_update(self.target_actors[i], self.actors[i], TAU)
            self.soft_update(self.target_critics[i], self.critics[i], TAU)

    def save_experience(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

# Main training loop
def main():
    maddpg = MADDPG()
    episodes = 1000
    max_steps = 100

    for episode in range(episodes):
        state = np.random.randn(NUM_AGENTS, STATE_DIM)  # Simulated environment
        episode_rewards = np.zeros(NUM_AGENTS)

        for t in range(max_steps):
            actions = maddpg.act(state)
            # Simulated environment step
            next_state = np.random.randn(NUM_AGENTS, STATE_DIM)
            rewards = np.random.randn(NUM_AGENTS)  # Simulated rewards
            done = np.random.rand() < 0.1  # Random termination

            maddpg.save_experience(state, actions, rewards, next_state, done)
            maddpg.learn()

            state = next_state
            episode_rewards += rewards

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards):.2f}")

if __name__ == "__main__":
    main()