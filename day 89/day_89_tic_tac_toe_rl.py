import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Constants
BOARD_SIZE = 9
NUM_ACTIONS = 9
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
EPSILON_START = 0.1
EPSILON_END = 0.01
EPSILON_DECAY = 1000
NUM_EPISODES = 10000
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network for Q-value approximation
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(BOARD_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )

    def forward(self, x):
        return self.model(x)

# Tic-Tac-Toe Environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros(BOARD_SIZE, dtype=np.float32)
        self.player = 1  # 1 for X, -1 for O

    def reset(self):
        self.board = np.zeros(BOARD_SIZE, dtype=np.float32)
        self.player = 1
        return self.board.copy()

    def get_valid_actions(self):
        return [i for i in range(BOARD_SIZE) if self.board[i] == 0]

    def make_move(self, action):
        if self.board[action] == 0:
            self.board[action] = self.player
            self.player = -self.player
            return True
        return False

    def check_game_over(self):
        # Check rows
        for i in range(0, 9, 3):
            if abs(self.board[i] + self.board[i+1] + self.board[i+2]) == 3:
                return self.board[i]
        # Check columns
        for i in range(3):
            if abs(self.board[i] + self.board[i+3] + self.board[i+6]) == 3:
                return self.board[i]
        # Check diagonals
        if abs(self.board[0] + self.board[4] + self.board[8]) == 3:
            return self.board[0]
        if abs(self.board[2] + self.board[4] + self.board[6]) == 3:
            return self.board[2]
        # Check draw
        if not any(self.board == 0):
            return 0
        return None

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state, dtype=np.float32), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# Training function
def train_dqn():
    env = TicTacToe()
    model = DQN().to(DEVICE)
    target_model = DQN().to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                valid_actions = env.get_valid_actions()
                action = random.choice(valid_actions) if valid_actions else random.randint(0, NUM_ACTIONS-1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(DEVICE)
                    q_values = model(state_tensor)
                    valid_actions = env.get_valid_actions()
                    q_values_valid = q_values[valid_actions]
                    action = valid_actions[torch.argmax(q_values_valid).item()] if valid_actions else random.randint(0, NUM_ACTIONS-1)

            # Make move
            if not env.make_move(action):
                reward = -1  # Penalty for invalid move
                done = True
            else:
                result = env.check_game_over()
                if result is not None:
                    reward = result if env.player == -1 else -result  # Reward from current player's perspective
                    done = True
                else:
                    reward = 0

            next_state = env.board.copy()
            episode_reward += reward

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Train model
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)

                # Compute Q-values
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    targets = rewards + (1 - dones) * DISCOUNT_FACTOR * next_q_values

                # Loss and optimization
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update target network
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-episode / EPSILON_DECAY)

        if episode % 1000 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

    # Save model
    torch.save(model.state_dict(), "tic_tac_toe_dqn.pt")
    print("Training completed. Model saved to tic_tac_toe_dqn.pt")

if __name__ == "__main__":
    train_dqn()