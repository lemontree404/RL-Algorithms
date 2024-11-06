import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

class GridWorld:
    """Grid World Environment"""
    def __init__(self, width: int, height: int, obstacles: List[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.action_names = ['right', 'down', 'left', 'up']
        
        # Random start and end points
        self.reset_start_end()
    
    def reset_start_end(self):
        """Randomly set start and end positions avoiding obstacles"""
        available_positions = [(x, y) for x in range(self.width) 
                             for y in range(self.height) 
                             if (x, y) not in self.obstacles]
        
        self.start = random.choice(available_positions)
        available_positions.remove(self.start)
        self.goal = random.choice(available_positions)
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is valid (within bounds and not an obstacle)"""
        x, y = state
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def get_next_state(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        """Get next state given current state and action"""
        next_x = state[0] + action[0]
        next_y = state[1] + action[1]
        
        if self.is_valid_state((next_x, next_y)):
            return (next_x, next_y)
        return state
    
    def get_reward(self, state: Tuple[int, int]) -> float:
        """Get reward for being in a state"""
        if state == self.goal:
            return 100.0
        return -1.0
    
    def plot_grid(self, path: List[Tuple[int, int]] = None, policy: np.ndarray = None):
        """Plot the grid with obstacles, start, goal, and optimal path"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set plot limits and aspect ratio
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal')
        
        # Remove axis ticks and grid
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot obstacles
        for x, y in self.obstacles:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='black'))

        # Plot policy arrows if policy is provided
        if policy is not None:
            path_set = set(path) if path else set()
            for x in range(self.width):
                for y in range(self.height):
                    if (x, y) not in path_set and (x, y) not in self.obstacles and (x, y) != self.goal:
                        action = self.actions[policy[y, x]]
                        dx, dy = action[0] * 0.3, action[1] * 0.3
                        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, 
                                fc='blue', ec='blue', alpha=0.5)

        # Plot start and goal
        ax.plot(self.start[0], self.start[1], 'gs', markersize=20, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'rd', markersize=20, label='Goal')

        # Plot path
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'y-', linewidth=3, zorder=3)

        ax.legend()
        plt.tight_layout()
        plt.show()