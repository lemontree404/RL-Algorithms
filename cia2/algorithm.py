from grid import GridWorld
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class PathFindingAlgorithm(ABC):
    """Abstract base class for path finding algorithms"""
    def __init__(self, env: GridWorld):
        self.env = env

class MDP(PathFindingAlgorithm):
    """Value Iteration algorithm for finding optimal path"""
    def __init__(self, env: GridWorld, discount: float = 0.99):
        super().__init__(env)
        self.discount = discount
        self.values = np.zeros((env.height, env.width))
        self.policy = np.zeros((env.height, env.width), dtype=int)
    
    def solve(self, epsilon: float = 0.001, max_iterations: int = 1000) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Perform value iteration to find optimal path"""
        iteration = 0
        delta = float('inf')
        
        while delta > epsilon and iteration < max_iterations:
            delta = 0
            new_values = np.zeros_like(self.values)
            
            for x in range(self.env.width):
                for y in range(self.env.height):
                    if (x, y) in self.env.obstacles or (x, y) == self.env.goal:
                        continue
                    
                    # Calculate value for each action
                    action_values = []
                    for action in self.env.actions:
                        next_state = self.env.get_next_state((x, y), action)
                        value = self.env.get_reward(next_state) + self.discount * self.values[next_state[1], next_state[0]]
                        action_values.append(value)
                    
                    # Update value and policy
                    new_values[y, x] = max(action_values)
                    self.policy[y, x] = np.argmax(action_values)
                    
                    delta = max(delta, abs(new_values[y, x] - self.values[y, x]))
            
            self.values = new_values
            iteration += 1
        
        return self.get_optimal_path(), self.policy
    
    def get_optimal_path(self) -> List[Tuple[int, int]]:
        """Get optimal path using current policy"""
        path = [self.env.start]
        current_state = self.env.start
        
        while current_state != self.env.goal:
            action = self.env.actions[self.policy[current_state[1], current_state[0]]]
            current_state = self.env.get_next_state(current_state, action)
            path.append(current_state)
            
            if len(path) > self.env.width * self.env.height:
                break
                
        return path

    def plot_value_function(self):
        """Plot the value function as a heatmap"""
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.values, cmap='viridis', origin='lower')
        plt.colorbar(im)
        ax.set_title('Value Function')
        plt.tight_layout()
        plt.show()

class QLearning(PathFindingAlgorithm):
    """Q-Learning algorithm for finding optimal path"""
    def __init__(self, env: GridWorld, discount: float = 0.99, alpha: float = 0.4, epsilon: float = 0.1):
        super().__init__(env)
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_values = np.zeros((env.height, env.width, len(env.actions)))

    def solve(self, max_episodes: int = 10000) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Perform Q-Learning to find optimal path"""
        for episode in range(max_episodes):
            state = self.env.start
            done = False

            while not done:
                # Choose action using epsilon-greedy policy
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(len(self.env.actions))
                else:
                    action = np.argmax(self.q_values[state[1], state[0]])

                # Take action and observe reward and next state
                next_state = self.env.get_next_state(state, self.env.actions[action])
                reward = self.env.get_reward(next_state)

                # Update Q-value
                self.q_values[state[1], state[0], action] += self.alpha * (
                    reward + self.discount * np.max(self.q_values[next_state[1], next_state[0]]) -
                    self.q_values[state[1], state[0], action]
                )

                # Check if goal is reached
                if next_state == self.env.goal:
                    done = True
                else:
                    state = next_state

        # Recover optimal path from Q-values
        path = self.get_optimal_path()
        policy = np.argmax(self.q_values, axis=2)
        return path, policy

    def get_optimal_path(self) -> List[Tuple[int, int]]:
        """Get optimal path using learned Q-values"""
        path = [self.env.start]
        current_state = self.env.start

        while current_state != self.env.goal:
            action = np.argmax(self.q_values[current_state[1], current_state[0]])
            current_state = self.env.get_next_state(current_state, self.env.actions[action])
            path.append(current_state)

            if len(path) > self.env.width * self.env.height:
                break

        return path