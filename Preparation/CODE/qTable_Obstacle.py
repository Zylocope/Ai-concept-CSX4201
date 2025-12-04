import numpy as np
import random
import matplotlib.pyplot as plt

class GridWorldEnv:
    """
    Simple GridWorld environment with obstacles and out-of-bound penalty.
    
    Layout (example 4x4):
      (row=0, col=0) is top-left (Start: S)
      (row=3, col=3) is bottom-right (Goal: G)

    Obstacles are defined as a list of (row, col) cells that the agent cannot enter.
    
    The agent receives +1 for reaching the goal.
    If the agent tries to move out-of-bound, it receives -1 and the episode terminates.
    If the agent tries to move into an obstacle cell, it stays in the same cell.
    
    max_steps is the maximum moves allowed per episode to avoid infinite loops.
    """

    def __init__(self, rows=3, cols=3, max_steps=50):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        
        # Define start and goal positions
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)  # bottom-right corner
        
        # Define obstacles (here: one obstacle at (1,1))
        self.obstacles = [(1, 1)]
        
        self.state = self.start
        self.step_count = 0
        
        self.num_states = rows * cols
        self.num_actions = 4  # 0: Up, 1: Down, 2: Left, 3: Right

    def reset(self):
        """
        Reset the environment to the start position.
        Returns the integer representation of the start state.
        """
        self.state = self.start
        self.step_count = 0
        return self._state_to_int(self.state)

    def step(self, action):
        """
        Take an action:
          0 -> Up, 1 -> Down, 2 -> Left, 3 -> Right

        Returns:
          next_state (int): integer representation of the new state
          reward (float): reward for the action
          done (bool): whether the episode has ended
        """
        r, c = self.state

        # Check if the action would move the agent out-of-bound.
        if (action == 0 and r == 0) or \
           (action == 1 and r == self.rows - 1) or \
           (action == 2 and c == 0) or \
           (action == 3 and c == self.cols - 1):
            # Out-of-bound action: assign penalty and terminate.
            reward = -1.0
            done = True
            return self._state_to_int(self.state), reward, done

        # Otherwise, compute the new position.
        if action == 0:
            new_r, new_c = r - 1, c
        elif action == 1:
            new_r, new_c = r + 1, c
        elif action == 2:
            new_r, new_c = r, c - 1
        elif action == 3:
            new_r, new_c = r, c + 1

        # Check if the new position is an obstacle.
        if (new_r, new_c) in self.obstacles:
            # # If it's an obstacle, the agent doesn't move.
            # new_r, new_c = r, c
            # reward = 0.0
            # Hit obstacle action: assign penalty and terminate.
            reward = -1.0
            done = True
            return self._state_to_int(self.state), reward, done
        else:
            reward = 0.0  # default reward for a valid move

        # Update the state and step count.
        self.state = (new_r, new_c)
        self.step_count += 1

        # If the agent reaches the goal, assign +1 reward.
        if self.state == self.goal:
            reward = 1.0

        done = (self.state == self.goal) or (self.step_count >= self.max_steps)
        return self._state_to_int(self.state), reward, done

    def _state_to_int(self, state):
        """
        Convert a (row, col) tuple to a single integer:
        state_index = row * number_of_columns + col
        """
        return state[0] * self.cols + state[1]

    def int_to_state(self, state_int):
        """
        Inverse of _state_to_int: converts an integer state back to (row, col)
        """
        return (state_int // self.cols, state_int % self.cols)


def q_learning_with_obstacle_demo():
    # Hyperparameters
    num_episodes = 4000
    learning_rate = 0.1
    gamma = 0.99 #(discount factor) determines the importance of future rewards.
    epsilon = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.2

    # Create the environment (using a 4x4 grid here)
    env = GridWorldEnv(rows=4, cols=4, max_steps=50)
    
    # Initialize Q-table of size [num_states x num_actions]
    Q = np.zeros((env.num_states, env.num_actions))

    # Logging rewards per episode
    episode_rewards = []

    # Q-learning loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, env.num_actions - 1)  # random exploration
            else:
                action = np.argmax(Q[state])  # choose best action

            next_state, reward, done = env.step(action)
            
            # Q-learning update rule
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            Q[state, action] += learning_rate * (td_target - Q[state, action])
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        episode_rewards.append(total_reward)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_last_100 = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Epsilon: {epsilon:.3f}, "
                  f"Avg Reward (last 100): {avg_last_100:.3f}")

    # Finished training
    print("\nTraining finished!")
    print("Trained Q-Table (reshaped for clarity):\n")
    Q_reshaped = Q.reshape(env.rows, env.cols, env.num_actions)
    for row_idx in range(env.rows):
        for col_idx in range(env.cols):
            print(f"Cell ({row_idx},{col_idx}): [Up, Down, Left, Right] = {Q_reshaped[row_idx, col_idx]}")
        print()

    # Plot the training rewards
    plt.plot(episode_rewards)
    plt.title("Q-Learning with Obstacles: Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward per Episode")
    plt.show()
    
    # Visualize the learned policy with a simulated walk
    visualize_policy_walk(Q, env)
    print("Final Q-Table:")
    print(Q)


def visualize_policy_walk(Q, env):
    """
    Using the trained Q-table, simulate an episode (always taking the best action)
    and print a visual representation of the path taken.
    """
    state = env.reset()
    done = False
    path = [env.int_to_state(state)]  # list of (row, col) positions

    max_sim_steps = env.max_steps
    steps = 0
    while not done and steps < max_sim_steps:
        action = np.argmax(Q[state])
        next_state, reward, done = env.step(action)
        state = next_state
        path.append(env.int_to_state(state))
        steps += 1

    # Create a grid for visualization
    grid = [[" " for _ in range(env.cols)] for _ in range(env.rows)]
    
    # Mark obstacles
    for obs in env.obstacles:
        grid[obs[0]][obs[1]] = "X"
    
    # Mark Start and Goal
    start = env.start
    goal = env.goal
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"

    # Mark the path taken with '*' if not Start or Goal.
    for pos in path:
        if pos != start and pos != goal and grid[pos[0]][pos[1]] == " ":
            grid[pos[0]][pos[1]] = "*"

    print("\nVisual Walk (Path taken by the agent using the learned Q-table):")
    for row in grid:
        print(" | ".join(row))
        print("-" * (env.cols * 4 - 1))


if __name__ == "__main__":
    q_learning_with_obstacle_demo()

