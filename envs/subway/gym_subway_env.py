import random
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SubwayEnv(gym.Env):
    """
    Minimal Subway Surfer–like env for DRL testing.
    Player moves between 3 lanes to avoid obstacles coming toward them.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    NUM_LANES = 3
    ENV_LENGTH = 20  # how far obstacles move
    MAX_STEPS = 500

    OBSTACLE_PROB = 0.2  # chance obstacle spawns in lane each step

    PLAYER_SPEED = 1.0

    def __init__(self, reward_mode: str = "survival", seed: Optional[int] = None, max_steps: int = MAX_STEPS, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Observations: [player_lane_norm, player_speed_norm, dist_lane0, dist_lane1, dist_lane2]
        high = np.array([1.0] * (2 + self.NUM_LANES), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.NUM_LANES)  # move to lane 0,1,2

        # Internal state
        self.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        self.player_lane = 1
        self.steps = 0
        self.score = 0

        # obstacles: list of dicts {"lane": int, "dist": float}
        self.obstacles = []

        # coverage tracking
        self.lane_changes = 0
        self.visited_lanes = set([self.player_lane])

        obs = self._get_obs()
        info = {"score": self.score, "unique_lanes": len(self.visited_lanes)}
        return obs, info

    def step(self, action: int):
        # --- Move player ---
        prev_lane = self.player_lane
        if action == 0:  # move left
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 2:  # move right
            self.player_lane = min(self.NUM_LANES - 1, self.player_lane + 1)
        # action == 1 => stay

        # Track metrics
        if self.player_lane != prev_lane:
            self.lane_changes += 1  # increment when lane actually changes

        self.visited_lanes.add(self.player_lane)

        # --- Move obstacles toward player ---
        for ob in self.obstacles:
            ob["dist"] -= self.PLAYER_SPEED

        # --- Check collisions ---
        terminated = any(ob["lane"] == self.player_lane and ob["dist"] <= 0 for ob in self.obstacles)

        # --- Remove passed obstacles and count score ---
        passed_obstacles = [ob for ob in self.obstacles if ob["dist"] <= 0]
        self.score += len([ob for ob in passed_obstacles if ob["lane"] != self.player_lane])
        # Keep only obstacles still in play
        self.obstacles = [ob for ob in self.obstacles if ob["dist"] > -5]

        # --- Spawn new obstacles randomly ---
        for lane in range(self.NUM_LANES):
            if self._rnd.random() < self.OBSTACLE_PROB:
                self.obstacles.append({"lane": lane, "dist": self.ENV_LENGTH})

        # --- Check max steps
        truncated = self.steps >= self.max_steps

        # --- Compute reward ---
        reward = 0.0
        if self.reward_mode == "survival":
            reward += 1.0  # alive bonus
            if action in [0, 2]:  # reward evasive moves
                reward += 0.3
        elif self.reward_mode == "coverage":
            reward += 0.5  # alive bonus
            if action in [0, 2]:
                reward += 1.0  # lane change bonus

        if terminated:
            reward -= 1.0

        self.steps += 1
        obs = self._get_obs()
        info = {
            "score": self.score,
            "unique_lanes": len(self.visited_lanes),
            "lane_changes": self.lane_changes
        }

        return obs, float(reward), bool(terminated), bool(truncated), info


    def _get_obs(self):
        lane_dists = [1.0] * self.NUM_LANES  # normalized distance to nearest obstacle in each lane
        for lane in range(self.NUM_LANES):
            obstacles_in_lane = [ob["dist"]/self.ENV_LENGTH for ob in self.obstacles if ob["lane"]==lane]
            if obstacles_in_lane:
                lane_dists[lane] = min(obstacles_in_lane)
        obs = np.array([self.player_lane / (self.NUM_LANES - 1), self.PLAYER_SPEED / 1.0, *lane_dists], dtype=np.float32)
        return obs
