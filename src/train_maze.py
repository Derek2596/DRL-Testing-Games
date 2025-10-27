import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from envs.grid_maze.gym_maze_env import MazeEnv


def make_env(persona="survivor", seed=7, model="ppo"):
    # Map persona to reward mode
    if persona == "survivor":
        reward_mode = "survival"
    elif persona == "explorer":
        reward_mode = "coverage"
    else:
        reward_mode = "survival"

    env = MazeEnv(reward_mode=reward_mode, seed=seed)

    # Log path for monitoring
    log_path = os.path.join("./logs/maze", f"{model}_{persona}_monitor.csv")
    env = Monitor(env, log_path)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo", "a2c"], default="ppo")
    parser.add_argument("--persona", type=str, choices=["survivor", "explorer"], default="survivor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--outdir", type=str, default="./models/maze")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Create environment
    env = make_env(persona=args.persona, seed=args.seed, model=args.algo)

    # Choose algorithm
    if args.algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    else:
        from stable_baselines3 import A2C
        model = A2C("MlpPolicy", env, verbose=1, seed=args.seed)

    # Logger for TensorBoard + stdout
    new_logger = configure(args.outdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Training {args.algo.upper()} agent for persona '{args.persona}'...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save model
    save_path = os.path.join(args.outdir, f"{args.algo}_{args.persona}")
    model.save(save_path)
    print(f"Saved model to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
