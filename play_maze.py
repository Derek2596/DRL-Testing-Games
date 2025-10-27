import pygame
import sys
import numpy as np
from envs.grid_maze.gym_maze_env import MazeEnv

# --- Settings ---
TILE_SIZE = 80
FPS = 10
COLORS = {
    "empty": (255, 255, 255),  # white
    "agent": (255, 50, 50),    # red
    "goal": (0, 200, 0),       # green
    "obstacle": (100, 100, 100) # gray
}

# --- Key mapping ---
KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_w: 0,
    pygame.K_DOWN: 1,
    pygame.K_s: 1,
    pygame.K_LEFT: 2,
    pygame.K_a: 2,
    pygame.K_RIGHT: 3,
    pygame.K_d: 3,
}

def draw_grid(screen, env):
    """Draw the maze grid."""
    grid = env._get_obs()
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            value = grid[r, c]
            if value == 1.0:
                color = COLORS["agent"]
            elif value == 0.5:
                color = COLORS["goal"]
            elif value == -1.0:
                color = COLORS["obstacle"]
            else:
                color = COLORS["empty"]
            pygame.draw.rect(
                screen, color,
                (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            )
            pygame.draw.rect(
                screen, (0, 0, 0),
                (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1
            )

def main():
    pygame.init()
    env = MazeEnv(render_mode=None, reward_mode="coverage", seed=42)
    obs, info = env.reset()

    screen_size = env.grid_size * TILE_SIZE
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("🧩 Grid Maze Game")

    clock = pygame.time.Clock()
    running = True
    action = None

    print("🎮 Controls: Arrow Keys or WASD to move | ESC to quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_TO_ACTION:
                    action = KEY_TO_ACTION[event.key]

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            action = None  # reset input each step

            if terminated:
                if env.agent_pos == env.goal_pos:
                    print("🎉 You reached the goal!")
                else:
                    print("💥 Collision (impossible in this version).")
                running = False
            elif truncated:
                print("⏱️ Max steps reached!")
                running = False

        # Draw current state
        screen.fill((0, 0, 0))
        draw_grid(screen, env)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    env.close()
    sys.exit()

if __name__ == "__main__":
    main()
