import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters
STATE_DIM = 4          # e.g., CartPole state dimension
HIDDEN_DIM = 64        # Hidden layer size
ACTION_DIM = 2         # e.g., CartPole actions (left, right)
MAX_EPISODES = 1000    # Number of episodes
MAX_STEPS = 200        # Max steps per episode
LEARNING_RATE = 0.001  # Learning rate
GAMMA = 0.99           # Discount factor
TRAJECTORIES = 32      # Number of parallel trajectories
SEED = 42              # Random seed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# Policy network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, ACTION_DIM)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Simulate environment (simplified CartPole-like)
def simulate_environment(states, actions, step):
    rewards = np.zeros(TRAJECTORIES)
    dones = np.zeros(TRAJECTORIES, dtype=bool)
    next_states = states.copy()
    
    for i in range(TRAJECTORIES):
        # Simplified reward: +1 for each step, -10 if "done"
        rewards[i] = 1.0
        if np.random.rand() < 0.05:  # Random termination
            dones[i] = True
            rewards[i] = -10.0
        # Update state (dummy update)
        next_states[i] += 0.01 * actions[i]
    
    return next_states, rewards, dones

# Main training loop
def main():
    # Initialize policy and optimizer
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    for episode in range(MAX_EPISODES):
        # Initialize trajectories
        states = np.zeros((TRAJECTORIES, STATE_DIM), dtype=np.float32)
        log_probs = []
        rewards = []
        dones = np.zeros(TRAJECTORIES, dtype=bool)
        
        # Collect trajectories
        for step in range(MAX_STEPS):
            # Convert states to tensor
            state_tensor = torch.from_numpy(states).to(device)
            
            # Forward pass
            probs = policy(state_tensor)
            
            # Sample actions
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            
            # Store log probabilities
            log_probs.append(log_prob)
            
            # Simulate environment
            next_states, step_rewards, step_dones = simulate_environment(states, actions.cpu().numpy(), step)
            rewards.append(step_rewards)
            states = next_states
            dones |= step_dones
            
            # Break if all trajectories are done
            if dones.all():
                break
        
        # Compute discounted returns
        returns = []
        for i in range(TRAJECTORIES):
            G = 0.0
            trajectory_returns = []
            for t in range(len(rewards) - 1, -1, -1):
                G = rewards[t][i] + GAMMA * G
                trajectory_returns.append(G)
            trajectory_returns.reverse()
            returns.append(trajectory_returns)
        
        # Convert returns to tensor
        returns = torch.tensor(returns, dtype=torch.float32, device=device).transpose(0, 1)
        
        # Compute policy gradient loss
        policy_loss = []
        for t in range(len(log_probs)):
            policy_loss.append(-log_probs[t] * returns[t])
        policy_loss = torch.cat(policy_loss).sum() / TRAJECTORIES
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Compute and print average reward
        total_reward = np.sum(rewards) / TRAJECTORIES
        print(f"Episode {episode}: Average Reward = {total_reward:.2f}")

if __name__ == "__main__":
    main()