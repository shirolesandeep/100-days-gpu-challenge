import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters
STATE_DIM = 64
D_MODEL = 64
NUM_HEADS = 8
NUM_LAYERS = 1
D_FF = 256
NUM_ACTIONS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
SEQ_LEN = 1  # Single state per sequence for simplicity

# Transformer-based RL Policy Network
class TransformerPolicy(nn.Module):
    def __init__(self, state_dim, d_model, num_heads, num_layers, d_ff, num_actions):
        super(TransformerPolicy, self).__init__()
        self.d_model = d_model
        
        # Embedding layer to project state to d_model
        self.embedding = nn.Linear(state_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Policy head
        self.policy_head = nn.Linear(d_model, num_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.TransformerEncoderLayer):
            for p in module.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
    
    def forward(self, states):
        # states: [batch_size, seq_len, state_dim]
        x = self.embedding(states)  # [batch_size, seq_len, d_model]
        x = self.transformer(x)     # [batch_size, seq_len, d_model]
        x = x[:, -1, :]             # Take last token: [batch_size, d_model]
        logits = self.policy_head(x)  # [batch_size, num_actions]
        return torch.softmax(logits, dim=-1)

# Policy Gradient (REINFORCE) Update
def compute_policy_gradient(policy, optimizer, states, actions, rewards):
    policy.train()
    optimizer.zero_grad()
    
    # Forward pass
    probs = policy(states)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    
    # Compute loss: -sum(G * log_prob)
    loss = -(log_probs * rewards).mean()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Main training loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize policy and optimizer
    policy = TransformerPolicy(
        state_dim=STATE_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_actions=NUM_ACTIONS
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Training loop (with dummy data)
    num_episodes = 100
    for episode in range(num_episodes):
        # Generate dummy states, actions, and rewards
        states = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM).to(device)
        actions = torch.randint(0, NUM_ACTIONS, (BATCH_SIZE,)).to(device)
        rewards = torch.randn(BATCH_SIZE).to(device)
        
        # Discounted rewards
        discounted_rewards = []
        for t in range(len(rewards)):
            G = 0
            for i, r in enumerate(rewards[t:]):
                G += (GAMMA ** i) * r
            discounted_rewards.append(G)
        discounted_rewards = torch.tensor(discounted_rewards, device=device)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Update policy
        loss = compute_policy_gradient(policy, optimizer, states, actions, discounted_rewards)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()