import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class MazeEnv(gym.Env):
    """
    Grid Maze environment with obstacles and a goal.
    The agent gets rewards based on persona type:
    - 'survivor': loses a small amount per step, rewarded for reaching the goal.
    - 'explorer': gains for visiting new tiles, rewarded for exploring more of the maze.
    Obstacles act as walls and cannot be entered.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, reward_mode="survival", seed=None, render_mode=None):
        super().__init__()
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self._rnd = np.random.default_rng(seed)
        self.seed_value = seed

        self.grid_size = 5
        self.num_actions = 4  # up, down, left, right
        self.max_steps = 50
        self.num_obstacles = 5

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # For rendering
        self.fig = None
        self.ax = None

        self.reset(seed=seed)

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rnd = np.random.default_rng(seed)
            self.seed_value = seed

        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]

        # Random obstacles, not on agent or goal
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            pos = (
                self._rnd.integers(0, self.grid_size),
                self._rnd.integers(0, self.grid_size),
            )
            if pos != tuple(self.agent_pos) and pos != tuple(self.goal_pos):
                self.obstacles.add(pos)

        self.steps = 0
        self.visited = {tuple(self.agent_pos)}

        obs = self._get_obs()
        info = {"score": 0}

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # Determine new potential position
        new_pos = list(self.agent_pos)
        if action == 0:  # up
            new_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            new_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            new_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            new_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)

        # Block movement into obstacles
        if tuple(new_pos) not in self.obstacles:
            self.agent_pos = new_pos
        else:
            reward -= 0.1

        pos_tuple = tuple(self.agent_pos)
        terminated = self.agent_pos == self.goal_pos
        truncated = self.steps >= self.max_steps

        # Reward logic
        if self.reward_mode == "survival":
            reward = -0.01  # small penalty per step
            if terminated:
                reward += 1.0 # win reward

        elif self.reward_mode == "coverage":
            reward = 0.0
            if pos_tuple not in self.visited:
                reward += 0.1  # reward for exploring new tiles
                self.visited.add(pos_tuple)
            if terminated:
                reward += 0.5 #small reward for winning

        else:
            reward = 0.0  # fallback

        obs = self._get_obs()
        info = {"score": int(terminated)}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[self.goal_pos[0], self.goal_pos[1]] = 0.5
        for (x, y) in self.obstacles:
            grid[x, y] = -1.0
        grid[self.agent_pos[0], self.agent_pos[1]] = 1.0
        return grid

    # ---------------- Rendering ----------------
    def render(self):
        grid = self._get_obs()

        cmap = colors.ListedColormap(["black", "white", "green", "red"])
        norm = colors.BoundaryNorm([-1.5, -0.1, 0.1, 0.6, 1.5], cmap.N)

        render_grid = np.zeros_like(grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = grid[i, j]
                if val == -1.0:
                    render_grid[i, j] = -1  # obstacle
                elif val == 0.5:
                    render_grid[i, j] = 0.5  # goal
                elif val == 1.0:
                    render_grid[i, j] = 1  # agent

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Maze Environment")
            self.ax.axis("off")

        self.ax.imshow(render_grid, cmap=cmap, norm=norm)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.pause(0.1)
        plt.draw()

        if self.render_mode == "rgb_array":
            # Convert to numpy array for gym rendering
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None