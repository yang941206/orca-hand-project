"""
SAC / TD3 速度實測(用來估算實驗矩陣要花多久),握拳 v2 任務。

off-policy 演算法每收一步資料就要做梯度更新,網路也比 PPO 大(256×256),
所以速度特性跟 PPO 不同,CPU/GPU 的結論也可能不同,要另外量。

設定說明:
- n_envs=1:SB3 預設(train_freq=1, gradient_steps=1),每 1 筆資料更新 1 次
- n_envs=8:gradient_steps=8,維持「每 1 筆資料更新 1 次」(UTD ratio = 1),
  才能跟 n_envs=1 公平比較,只差在環境模擬有沒有平行
"""

import argparse
import json
import time

import numpy as np
import torch
from gymnasium.wrappers import RescaleAction, TimeLimit
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from fist_task_v2 import OrcaFistTaskV2

ALGOS = {"sac": SAC, "td3": TD3}


def make_env():
    def _init():
        env = OrcaFistTaskV2()
        ones = np.ones(env.action_space.shape, dtype=np.float32)
        return TimeLimit(RescaleAction(env, -ones, ones), max_episode_steps=300)
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS), default="sac")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=10_000)
    args = parser.parse_args()

    fns = [make_env() for _ in range(args.n_envs)]
    env = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)
    model = ALGOS[args.algo]("MlpPolicy", env, device=args.device, seed=0, verbose=0,
                             learning_starts=1000, gradient_steps=args.n_envs)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    if args.device == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    env.close()
    print(json.dumps({"algo": args.algo, "device": args.device, "n_envs": args.n_envs,
                      "timesteps": model.num_timesteps, "learn_s": round(sec, 1),
                      "fps": round(model.num_timesteps / sec, 1)}))
