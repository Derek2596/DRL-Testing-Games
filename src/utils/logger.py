# src/utils/logger.py
import csv
import json
from pathlib import Path
from typing import List, Dict

class MetricsLogger:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = []

    def log_episode(self, metrics: Dict):
        self.episodes.append(metrics)

    def save_csv(self, filename="episodes.csv"):
        keys = sorted({k for e in self.episodes for k in e.keys()})
        with open(self.out_dir / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for e in self.episodes:
                writer.writerow(e)

    def save_json(self, filename="episodes.json"):
        with open(self.out_dir / filename, "w") as f:
            json.dump(self.episodes, f, indent=2)
