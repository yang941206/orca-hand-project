"""
CPU vs GPU 小規模實測:用捏合任務跑 PPO,真的計時(wall-clock)。

用法:
    python bench_device.py --device cpu  --n-envs 8
    python bench_device.py --device cuda --n-envs 8

會印出:
- total  : 從建立環境到訓練結束的總時間(包含 CUDA 初始化等開銷)
- learn  : 只算 model.learn() 的時間
"""

import argparse
import json
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers import TimeLimit

from pinch_task import OrcaPinchTask


def make_env():
    def _init():
        return TimeLimit(OrcaPinchTask(), max_episode_steps=300)
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    fns = [make_env() for _ in range(args.n_envs)]
    env = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)

    model = PPO("MlpPolicy", env, device=args.device, seed=args.seed, verbose=0)
    t1 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    if args.device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    env.close()

    result = {
        "device": args.device,
        "n_envs": args.n_envs,
        "timesteps": model.num_timesteps,
        "total_s": round(t2 - t0, 2),
        "learn_s": round(t2 - t1, 2),
        "fps": round(model.num_timesteps / (t2 - t1), 1),
    }
    print(json.dumps(result))
