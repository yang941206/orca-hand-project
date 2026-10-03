"""
開 MuJoCo 視窗觀看框架訓練出來的任何一個模型。會讀該實驗的 config.json,用相同的任務設定。

用法:
    python watch_experiment.py results/experiments/fist/ppo/default_seed0
    python watch_experiment.py results/experiments/pinch/sac/default_seed0 --episodes 5 --stochastic
"""

import argparse
import json
import os
import time

import numpy as np
from gymnasium.wrappers import RescaleAction

from rl_framework import ALGOS, TASKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--stochastic", action="store_true", help="用隨機策略(預設 deterministic)")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = ALGOS[cfg["algo"]].load(os.path.join(args.run_dir, "model"), device="cpu")

    env = TASKS[cfg["task"]](render_mode="human", reset_noise=cfg["reset_noise"])
    ones = np.ones(env.action_space.shape, dtype=np.float32)
    env = RescaleAction(env, -ones, ones)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        for t in range(1, cfg["max_episode_steps"] + 1):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(1.0 / env.unwrapped.metadata["render_fps"])
            if terminated or truncated:
                break
        print(f"第 {ep + 1} 回合:{'成功' if info.get('is_success') else '未成功'}(第 {t} 步)")
        time.sleep(1.0)
    env.close()
