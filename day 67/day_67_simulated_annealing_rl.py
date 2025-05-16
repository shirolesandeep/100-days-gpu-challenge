import torch
import numpy as np

# Constants for the RL environment
GRID_SIZE = 5
NUM_ACTIONS = 4  # Up, Down, Left, Right
NUM_STATES = GRID_SIZE * GRID_SIZE  # 25 states
MAX_STEPS = 100  # Max steps per episode
NUM_CANDIDATES = 1024  # Number of parallel candidate policies
MAX_ITERATIONS = 1000  # Number of SA iterations
INITIAL_TEMP = 10.0  # Initial temperature
COOLING_RATE = 0.995  # Temperature reduction factor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(policies, num_candidates):
    """
    Evaluate the fitness (cumulative reward) of multiple policies.
    policies: Tensor of shape [num_candidates, num_states]
    """
    batch_size = num_candidates
    states = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)  # Start at state 0
    rewards = torch.zeros(batch_size, device=DEVICE)
    steps = 0

    # Define environment
    goal_state = NUM_STATES - 1  # Bottom-right
    goal_reward = 100.0
    step_cost = -1.0
    obstacle_penalty = -10.0
    obstacles = torch.tensor([6, 8, 12, 16, 18], dtype=torch.long, device=DEVICE)

    active = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)  # Track active episodes

    while steps < MAX_STEPS and active.any():
        # Get actions for current states
        actions = policies[torch.arange(batch_size, device=DEVICE), states]

        # Compute next states
        rows = states // GRID_SIZE
        cols = states % GRID_SIZE
        next_states = states.clone()

        # Apply actions (Up: 0, Down: 1, Left: 2, Right: 3)
        up_mask = (actions == 0) & (rows > 0)
        down_mask = (actions == 1) & (rows < GRID_SIZE - 1)
        left_mask = (actions == 2) & (cols > 0)
        right_mask = (actions == 3) & (cols < GRID_SIZE - 1)

        next_states[up_mask] = states[up_mask] - GRID_SIZE
        next_states[down_mask] = states[down_mask] + GRID_SIZE
        next_states[left_mask] = states[left_mask] - 1
        next_states[right_mask] = states[right_mask] + 1

        # Compute rewards
        step_rewards = torch.zeros(batch_size, device=DEVICE)
        goal_mask = (next_states == goal_state) & active
        obstacle_mask = torch.isin(next_states, obstacles) & active
        step_mask = active & ~goal_mask & ~obstacle_mask

        step_rewards[goal_mask] = goal_reward
        step_rewards[obstacle_mask] = obstacle_penalty
        step_rewards[step_mask] = step_cost

        rewards += step_rewards

        # Update active episodes
        active = active & ~goal_mask & ~obstacle_mask
        states[active] = next_states[active]
        steps += 1

    return rewards

def simulated_annealing():
    # Initialize policies randomly
    current_policies = torch.randint(0, NUM_ACTIONS, (NUM_CANDIDATES, NUM_STATES), device=DEVICE)
    current_fitness = evaluate_policy(current_policies, NUM_CANDIDATES)
    best_policies = current_policies.clone()
    best_fitness = current_fitness.clone()

    temperature = INITIAL_TEMP

    for iteration in range(MAX_ITERATIONS):
        # Perturb policies: randomly change one action per policy
        new_policies = current_policies.clone()
        states_to_change = torch.randint(0, NUM_STATES, (NUM_CANDIDATES,), device=DEVICE)
        new_actions = torch.randint(0, NUM_ACTIONS, (NUM_CANDIDATES,), device=DEVICE)
        new_policies[torch.arange(NUM_CANDIDATES), states_to_change] = new_actions

        # Evaluate new policies
        new_fitness = evaluate_policy(new_policies, NUM_CANDIDATES)

        # Compute acceptance probabilities
        delta = new_fitness - current_fitness
        accept_probs = torch.where(
            delta > 0,
            torch.ones_like(delta),
            torch.exp(delta / temperature)
        )
        rand_vals = torch.rand(NUM_CANDIDATES, device=DEVICE)

        # Accept or reject new policies
        accept_mask = rand_vals < accept_probs
        current_policies[accept_mask] = new_policies[accept_mask]
        current_fitness[accept_mask] = new_fitness[accept_mask]

        # Update best policies
        better_mask = new_fitness > best_fitness
        best_policies[better_mask] = new_policies[better_mask]
        best_fitness[better_mask] = new_fitness[better_mask]

        # Cool down temperature
        temperature *= COOLING_RATE

        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Avg Fitness: {best_fitness.mean().item():.2f}, Temp: {temperature:.2f}")

    # Return the best policy
    best_idx = torch.argmax(best_fitness)
    return best_policies[best_idx], best_fitness[best_idx]

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(42)

    # Run simulated annealing
    best_policy, best_fitness = simulated_annealing()
    print(f"Best Policy Fitness: {best_fitness.item():.2f}")
    print("Best Policy (action per state):")
    print(best_policy.cpu().numpy())

if __name__ == "__main__":
    main()