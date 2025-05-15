import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim=1, device='cuda'):
        """
        Initialize replay buffer with CUDA tensors.
        
        Args:
            capacity (int): Maximum number of experiences to store
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space (default=1 for discrete actions)
            device (str): Device to store tensors ('cuda' or 'cpu')
        """
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize buffers as CUDA tensors
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.int64, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.size = 0
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int or np.ndarray): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Terminal state indicator
        """
        # Convert inputs to tensors if necessary
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        # Store experience
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of experiences using CUDA-accelerated random sampling.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            raise ValueError("Not enough experiences in buffer to sample")

        # Generate random indices on GPU
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Gather samples using advanced indexing
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer."""
        return self.size

# Example usage
if __name__ == "__main__":
    # Initialize parameters
    capacity = 100000
    state_dim = 4
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create replay buffer
    buffer = ReplayBuffer(capacity, state_dim, device=device)

    # Simulate adding experiences
    for _ in range(1000):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, 2)
        reward = np.random.random()
        next_state = np.random.randn(state_dim)
        done = np.random.choice([True, False])
        buffer.push(state, action, reward, next_state, done)

    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Print shapes and device info
    print(f"States shape: {states.shape}, device: {states.device}")
    print(f"Actions shape: {actions.shape}, device: {actions.device}")
    print(f"Rewards shape: {rewards.shape}, device: {rewards.device}")
    print(f"Next states shape: {next_states.shape}, device: {next_states.device}")
    print(f"Dones shape: {dones.shape}, device: {dones.device}")