import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from envs.subway.gym_subway_env import SubwayEnv

def make_env(persona="survivor", seed=7, model="a2c"):
    # Map persona to reward_mode
    if persona == "survivor":
        reward_mode = "survival"
    elif persona == "explorer":
        reward_mode = "coverage"
    else:
        reward_mode = "survival"

    env = SubwayEnv(reward_mode=reward_mode, seed=seed)

    log_path = os.path.join("./logs/subway", f"{model}_{persona}_monitor.csv")
    env = Monitor(env, log_path)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo","a2c"], default="ppo")
    parser.add_argument("--persona", type=str, choices=["survivor","explorer"], default="survivor")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--outdir", type=str, default="./models/subway")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    env = make_env(persona=args.persona, seed=args.seed, model=args.algo)

    if args.algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    else:
        from stable_baselines3 import A2C
        model = A2C("MlpPolicy", env, verbose=1, seed=args.seed)

    # Logger for Tensorboard
    new_logger = configure(args.outdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    save_path = os.path.join(args.outdir, f"{args.algo}_{args.persona}")
    model.save(save_path)
    print(f"Saved model to {save_path}")

    env.close()

if __name__ == "__main__":
    main()
