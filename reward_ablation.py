"""
握拳 reward 設計小實驗(ablation):三種 reward 用 PPO 實際訓練,比較成功率。

三個版本只差 reward 的距離項係數,成功判定(物理條件)完全相同:
- pose            :只有加權姿勢誤差
- pose_tip        :+ 四指指尖到掌心距離(正式版)
- pose_tip_thumb  :+ 四指指尖到掌心距離 + 拇指到食/中指距離

用法:
    python reward_ablation.py                       # 全部版本 × 2 seed
    python reward_ablation.py --timesteps 100000    # 先用小步數試跑
    python reward_ablation.py --timesteps 1000000 --variants pose --out-dir results/reward_ablation_1M
結果存到 results/reward_ablation/
"""

import argparse
import json
import os
import time

import numpy as np
from gymnasium.wrappers import RescaleAction, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from fist_task_v2 import OrcaFistTaskV2

OUT_DIR = "results/reward_ablation"  # 可用 --out-dir 覆蓋
MAX_EPISODE_STEPS = 300


class FistPoseOnly(OrcaFistTaskV2):
    TIP_DIST_COEF = 0.0
    THUMB_DIST_COEF = 0.0


class FistPoseTip(OrcaFistTaskV2):
    TIP_DIST_COEF = 1.0
    THUMB_DIST_COEF = 0.0


class FistPoseTipThumb(OrcaFistTaskV2):
    TIP_DIST_COEF = 1.0
    THUMB_DIST_COEF = 1.0


VARIANTS = {
    "pose": FistPoseOnly,
    "pose_tip": FistPoseTip,
    "pose_tip_thumb": FistPoseTipThumb,
}


def make_env(env_cls):
    def _init():
        env = env_cls()
        ones = np.ones(env.action_space.shape, dtype=np.float32)
        env = RescaleAction(env, -ones, ones)  # 動作正規化到 [-1, 1]
        return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return _init


class SuccessCurveCallback(BaseCallback):
    """每次 rollout 結束記錄:目前步數、最近 100 個 episode 的成功率與平均長度。"""

    def __init__(self):
        super().__init__()
        self.curve = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_success_buffer
        lens = [ep["l"] for ep in self.model.ep_info_buffer]
        rews = [ep["r"] for ep in self.model.ep_info_buffer]
        self.curve.append({
            "timesteps": self.num_timesteps,
            "success_rate": float(np.mean(buf)) if len(buf) else 0.0,
            "ep_len_mean": float(np.mean(lens)) if lens else None,
            "ep_rew_mean": float(np.mean(rews)) if rews else None,
        })


def evaluate(model, env_cls, n_stochastic: int = 20):
    """環境沒有隨機性,所以 deterministic 只需跑 1 集;另外跑 n 集隨機策略看穩健度。"""
    env = make_env(env_cls)()
    results = {}
    for mode, n in [("deterministic", 1), ("stochastic", n_stochastic)]:
        successes, steps_to_success = [], []
        for ep in range(n):
            obs, _ = env.reset(seed=ep)
            for t in range(1, MAX_EPISODE_STEPS + 1):
                action, _ = model.predict(obs, deterministic=(mode == "deterministic"))
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            successes.append(bool(info.get("is_success", False)))
            if successes[-1]:
                steps_to_success.append(t)
        results[mode] = {
            "success_rate": float(np.mean(successes)),
            "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else None,
        }
    env.close()
    return results


def run_one(variant: str, seed: int, timesteps: int, n_envs: int):
    env_cls = VARIANTS[variant]
    env = VecMonitor(SubprocVecEnv([make_env(env_cls) for _ in range(n_envs)]))
    model = PPO("MlpPolicy", env, device="cpu", seed=seed, verbose=0)
    callback = SuccessCurveCallback()

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callback)
    train_s = time.perf_counter() - t0
    env.close()

    result = {
        "variant": variant, "seed": seed, "timesteps": model.num_timesteps,
        "train_seconds": round(train_s, 1),
        "eval": evaluate(model, env_cls),
        "curve": callback.curve,
    }
    model.save(os.path.join(OUT_DIR, f"{variant}_seed{seed}"))
    with open(os.path.join(OUT_DIR, f"{variant}_seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    total = len(args.variants) * len(args.seeds)
    k = 0
    for variant in args.variants:
        for seed in args.seeds:
            k += 1
            r = run_one(variant, seed, args.timesteps, args.n_envs)
            last = r["curve"][-1] if r["curve"] else {}
            print(f"[{k}/{total}] {variant:15s} seed={seed} 訓練 {r['train_seconds']:.0f}s | "
                  f"訓練末期成功率 {last.get('success_rate', 0) * 100:5.1f}% | "
                  f"ep_rew {last.get('ep_rew_mean') or 0:8.1f} | "
                  f"評估 det={r['eval']['deterministic']['success_rate'] * 100:.0f}% "
                  f"stoch={r['eval']['stochastic']['success_rate'] * 100:.0f}% "
                  f"平均成功步數 {r['eval']['stochastic']['mean_steps_to_success']}", flush=True)
