import argparse
import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from envs.grid_maze.gym_maze_env import MazeEnv


def run_episode(model, env, render=False, delay=0.3):
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    visited = set()

    if render:
        env.render()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        visited.add(tuple(env.agent_pos))
        done = terminated or truncated

        if render:
            env.render()
            time.sleep(delay)

    coverage = len(visited)
    return total_reward, steps, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--persona", choices=["survivor", "explorer"], default="survivor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--modeldir", default="./models/maze")
    parser.add_argument("--outdir", default="./logs/maze")
    parser.add_argument("--render", action="store_true", help="Render a single episode visually")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load environment with same persona logic
    reward_mode = "survival" if args.persona == "survivor" else "coverage"
    env = MazeEnv(
        reward_mode=reward_mode,
        seed=args.seed,
        render_mode="human" if args.render else None
    )

    # Load trained model
    model_path = os.path.join(args.modeldir, f"{args.algo}_{args.persona}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        model = A2C.load(model_path, env=env)

    print(f"Evaluating {args.algo.upper()} model for '{args.persona}' on MazeEnv")

    # If render mode, only run one episode visually
    if args.render:
        print("Running one rendered episode...")
        total_reward, steps, coverage = run_episode(model, env, render=True)
        print(f"\nRendered Episode | Reward: {total_reward:.2f} | Steps: {steps} | Coverage: {coverage}")
        env.close()
        return

    # Run normal evaluation for metrics
    results = []
    for ep in range(args.episodes):
        total_reward, steps, coverage = run_episode(model, env)
        results.append({
            "episode": ep + 1,
            "reward": total_reward,
            "steps": steps,
            "coverage": coverage
        })

    # Save evaluation results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.outdir, f"{args.algo}_{args.persona}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")

    print("\nSummary:")
    print(df.describe())

    env.close()


if __name__ == "__main__":
    main()