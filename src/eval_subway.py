# Evaluates trained SubwayEnv models and logs gameplay metrics
import argparse, os, csv
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.subway.gym_subway_env import SubwayEnv


def run_episode(model, persona="survivor", render=False):
    """
    Run one evaluation episode with a trained model.
    Returns per-episode gameplay metrics.
    """
    env = SubwayEnv(render_mode="human" if render else None)
    obs, info = env.reset()
    done = trunc = False

    ep_reward = 0.0
    steps = 0
    lane_changes = 0  # number of times agent moved left/right

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        lane_changes += int(action in [0, 2])  # assuming 0=left, 2=right
        obs, r, done, trunc, info = env.step(int(action))
        ep_reward += r
        steps += 1

    # Collect metrics from env info
    score = int(info.get("score", 0))
    unique_lanes = int(info.get("unique_lanes", 0))  # for explorer persona

    env.close()
    return {
        "reward": float(ep_reward),
        "score": score,
        "steps": steps,
        "lane_changes": lane_changes,
        "terminated": int(done),
        "truncated": int(trunc),
        "unique_lanes": unique_lanes,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modeldir", type=str, default="./models/subway")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", type=int, default=0)
    args = p.parse_args()

    # --- Auto-detect model type and persona from path ---
    base = os.path.basename(args.model_path).lower()
    if "ppo" in base:
        algo = "ppo"
        ModelClass = PPO
    elif "a2c" in base:
        algo = "a2c"
        ModelClass = A2C
    else:
        raise ValueError("Model path must include 'ppo' or 'a2c' to determine algorithm.")

    if "survivor" in base:
        persona = "survivor"
    elif "explorer" in base:
        persona = "explorer"
    else:
        raise ValueError("Model path must include 'survivor' or 'explorer' to determine persona.")

    # --- Auto-generate CSV output filename ---
    os.makedirs("logs/subway", exist_ok=True)
    csv_out = f"logs/subway/{algo}_{persona}_metrics.csv"

    # --- Load model ---
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    model = ModelClass.load(args.model_path)

    # --- Run episodes ---
    rows = []
    for ep in range(1, args.episodes + 1):
        metrics = run_episode(model, persona=persona, render=bool(args.render))
        metrics["episode"] = ep
        rows.append(metrics)

    # --- Aggregate metrics ---
    mean_reward = np.mean([r["reward"] for r in rows])
    std_reward = np.std([r["reward"] for r in rows])
    mean_score = np.mean([r["score"] for r in rows])
    mean_lane_changes = np.mean([r["lane_changes"] for r in rows])
    mean_lanes = np.mean([r["unique_lanes"] for r in rows])
    crash_rate = np.mean([r["terminated"] for r in rows])

    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean score: {mean_score:.2f}")
    print(f"Mean lane changes: {mean_lane_changes:.2f}")
    print(f"Mean unique lanes: {mean_lanes:.2f}")
    print(f"Crash rate: {crash_rate * 100:.1f}%")
    print(f"Saved per-episode metrics to: {csv_out}")

    # --- Save CSV ---
    fieldnames = [
        "episode", "reward", "score", "steps",
        "lane_changes", "terminated", "truncated", "unique_lanes"
    ]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
