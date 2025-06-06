import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import uuid

# Constants
NUM_AGENTS = 4
OBS_DIM = 10
ACTION_DIM = 5
HIDDEN_DIM = 64
STATE_DIM = 20
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.01
LR = 0.0005
MEMORY_SIZE = 10000
EPISODES = 1000

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DRQN Agent Network
class DRQNAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(DRQNAgent, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.hidden_dim = hidden_dim

    def forward(self, obs, prev_action, hidden):
        x = torch.cat([obs, prev_action], dim=-1)
        h = torch.relu(self.fc1(x))
        h = self.gru(h, hidden)
        q_values = self.fc2(h)
        return q_values, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(device)

# QMIX Mixing Network
class QMIXMixer(nn.Module):
    def __init__(self, state_dim, num_agents, hidden_dim, action_dim):
        super(QMIXMixer, self).__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Hypernetworks for generating weights
        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * num_agents)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim * action_dim)
        self.hyper_b2 = nn.Linear(state_dim, action_dim)

    def forward(self, agent_qs, state):
        batch_size = state.size(0)
        # Reshape agent Q-values: [batch_size, num_agents, action_dim]
        agent_qs = agent_qs.view(batch_size, self.num_agents, self.action_dim)
        
        # First layer weights and biases
        w1 = torch.abs(self.hyper_w1(state))  # Ensure non-negative for monotonicity
        b1 = self.hyper_b1(state)
        hidden = torch.relu(torch.bmm(agent_qs, w1.view(batch_size, self.num_agents, self.hidden_dim)) + b1.unsqueeze(1))
        
        # Second layer weights and biases
        w2 = torch.abs(self.hyper_w2(state))  # Ensure non-negative for monotonicity
        b2 = self.hyper_b2(state)
        q_tot = torch.bmm(hidden, w2.view(batch_size, self.hidden_dim, self.action_dim)) + b2.unsqueeze(1)
        
        return q_tot.squeeze(1)  # [batch_size, action_dim]

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# QMIX Agent
class QMIX:
    def __init__(self):
        self.agents = [DRQNAgent(OBS_DIM, ACTION_DIM, HIDDEN_DIM).to(device) for _ in range(NUM_AGENTS)]
        self.mixer = QMIXMixer(STATE_DIM, NUM_AGENTS, HIDDEN_DIM, ACTION_DIM).to(device)
        self.target_agents = [DRQNAgent(OBS_DIM, ACTION_DIM, HIDDEN_DIM).to(device) for _ in range(NUM_AGENTS)]
        self.target_mixer = QMIXMixer(STATE_DIM, NUM_AGENTS, HIDDEN_DIM, ACTION_DIM).to(device)
        
        # Copy weights to target networks
        for agent, target_agent in zip(self.agents, self.target_agents):
            target_agent.load_state_dict(agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        self.optimizer = optim.Adam(
            list(self.mixer.parameters()) + [p for agent in self.agents for p in agent.parameters()],
            lr=LR
        )
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def act(self, obs, prev_actions, hiddens, epsilon):
        actions = []
        new_hiddens = []
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        prev_actions = torch.tensor(prev_actions, dtype=torch.float32).to(device)
        
        for i, (agent, hidden) in enumerate(zip(self.agents, hiddens)):
            q_values, h = agent(obs[:, i], prev_actions[:, i], hidden)
            if random.random() < epsilon:
                action = random.randint(0, ACTION_DIM - 1)
            else:
                action = q_values.argmax(dim=-1).item()
            actions.append(action)
            new_hiddens.append(h)
        
        return actions, new_hiddens

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        
        states = torch.tensor(batch[0], dtype=torch.float32).to(device)
        actions = torch.tensor(batch[1], dtype=torch.int64).to(device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).to(device)
        next_states = torch.tensor(batch[3], dtype=torch.float32).to(device)
        dones = torch.tensor(batch[4], dtype=torch.float32).to(device)
        obs = torch.tensor(batch[5], dtype=torch.float32).to(device)
        next_obs = torch.tensor(batch[6], dtype=torch.float32).to(device)
        prev_actions = torch.tensor(batch[7], dtype=torch.float32).to(device)
        
        # Compute current Q-values
        hiddens = [agent.init_hidden(BATCH_SIZE) for agent in self.agents]
        agent_qs = []
        for i, agent in enumerate(self.agents):
            q, h = agent(obs[:, :, i], prev_actions[:, :, i], hiddens[i])
            agent_qs.append(q.unsqueeze(1))
        agent_qs = torch.cat(agent_qs, dim=1)  # [batch_size, num_agents, action_dim]
        q_tot = self.mixer(agent_qs, states)
        q_tot_selected = q_tot.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        target_hiddens = [agent.init_hidden(BATCH_SIZE) for agent in self.target_agents]
        target_agent_qs = []
        for i, agent in enumerate(self.target_agents):
            q, h = agent(next_obs[:, :, i], torch.zeros_like(prev_actions[:, :, i]), target_hiddens[i])
            target_agent_qs.append(q.unsqueeze(1))
        target_agent_qs = torch.cat(target_agent_qs, dim=1)
        target_q_tot = self.target_mixer(target_agent_qs, next_states)
        target_max_q = target_q_tot.max(dim=1)[0]
        targets = rewards + GAMMA * (1 - dones) * target_max_q
        
        # Compute loss
        loss = nn.MSELoss()(q_tot_selected, targets.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target networks
        for agent, target_agent in zip(self.agents, self.target_agents):
            for param, target_param in zip(agent.parameters(), target_agent.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def main():
    # Initialize QMIX
    qmix = QMIX()
    
    # Dummy environment (replace with actual environment, e.g., SMAC)
    def dummy_env():
        obs = np.random.randn(BATCH_SIZE, NUM_AGENTS, OBS_DIM)
        state = np.random.randn(BATCH_SIZE, STATE_DIM)
        reward = np.random.randn(BATCH_SIZE)
        done = np.random.randint(0, 2, BATCH_SIZE)
        next_obs = np.random.randn(BATCH_SIZE, NUM_AGENTS, OBS_DIM)
        next_state = np.random.randn(BATCH_SIZE, STATE_DIM)
        return obs, state, reward, done, next_obs, next_state
    
    # Training loop
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999
    
    for episode in range(EPISODES):
        obs, state, _, _, _, _ = dummy_env()
        hiddens = [agent.init_hidden(1) for agent in qmix.agents]
        prev_actions = np.zeros((1, NUM_AGENTS, ACTION_DIM))
        episode_reward = 0
        
        done = False
        while not done:
            actions, new_hiddens = qmix.act(obs, prev_actions, hiddens, epsilon)
            _, _, reward, done, next_obs, next_state = dummy_env()
            
            # Store transition
            action_one_hot = np.zeros((1, NUM_AGENTS, ACTION_DIM))
            for i, a in enumerate(actions):
                action_one_hot[0, i, a] = 1.0
            qmix.memory.push((state, actions, reward, next_state, done, obs, next_obs, prev_actions))
            
            # Update
            qmix.update()
            
            # Update state
            obs = next_obs
            state = next_state
            hiddens = new_hiddens
            prev_actions = action_one_hot
            episode_reward += reward.mean()
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    main()