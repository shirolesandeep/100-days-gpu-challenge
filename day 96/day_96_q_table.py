import torch
import numpy as np

# Configuration
GRID_SIZE = 32
NUM_ACTIONS = 4
MAX_EPISODES = 1000
MAX_STEPS = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridWorld:
    def __init__(self):
        self.start_x, self.start_y = 0, 0
        self.goal_x, self.goal_y = GRID_SIZE - 1, GRID_SIZE - 1
        self.obstacles = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.obstacles[GRID_SIZE // 2, GRID_SIZE // 2] = 1  # Example obstacle
        self.obstacles = torch.tensor(self.obstacles, dtype=torch.int32, device=DEVICE)

    def get_reward_and_next_state(self, x, y, action):
        next_x, next_y = x, y
        if action == 0:  # Up
            next_x = x - 1
        elif action == 1:  # Down
            next_x = x + 1
        elif action == 2:  # Left
            next_y = y - 1
        elif action == 3:  # Right
            next_y = y + 1

        # Boundary check
        if next_x < 0 or next_x >= GRID_SIZE or next_y < 0 or next_y >= GRID_SIZE:
            return -1.0, x, y

        # Obstacle check
        if self.obstacles[next_x, next_y] == 1:
            return -1.0, x, y

        # Goal check
        if next_x == self.goal_x and next_y == self.goal_y:
            return 1.0, next_x, next_y
        return -0.01, next_x, next_y

def select_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    q_values = q_table[state]
    return torch.argmax(q_values).item()

def main():
    # Initialize environment and Q-table
    env = GridWorld()
    q_table = torch.zeros((GRID_SIZE * GRID_SIZE, NUM_ACTIONS), dtype=torch.float32, device=DEVICE)

    # Training loop
    for episode in range(MAX_EPISODES):
        x, y = env.start_x, env.start_y
        state = x * GRID_SIZE + y

        for step in range(MAX_STEPS):
            # Select action
            action = select_action(q_table, state, EPSILON)

            # Get next state and reward
            reward, next_x, next_y = env.get_reward_and_next_state(x, y, action)
            next_state = next_x * GRID_SIZE + next_y

            # Update Q-table
            with torch.no_grad():
                max_next_q = torch.max(q_table[next_state])
                q_table[state, action] += ALPHA * (reward + GAMMA * max_next_q - q_table[state, action])

            x, y = next_x, next_y
            state = next_state

            if x == env.goal_x and y == env.goal_y:
                break

        if episode % 100 == 0:
            print(f"Episode {episode} completed")

    print("Training completed!")
    return q_table

if __name__ == "__main__":
    q_table = main()
    # Optional: Save Q-table
    torch.save(q_table, "q_table.pt")
