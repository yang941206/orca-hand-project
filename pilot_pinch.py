"""
捏合 v2 可學性試驗:PPO 預設超參數,確認 RL 學得會,再排正式實驗矩陣。
訓練/評估流程跟 reward_ablation.py 完全相同(共用它的函式),只換任務。

用法:
    python pilot_pinch.py                                   # E13:無雜訊、100 萬步、seed 0 1
    python pilot_pinch.py --reset-noise 0.05 --timesteps 500000 --seeds 0 --out-dir results/noise_check
"""

import argparse
import os

import reward_ablation
from pinch_task_v2 import OrcaPinchTaskV2

reward_ablation.VARIANTS = {"pinch_v2": OrcaPinchTaskV2}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--reset-noise", type=float, default=0.0)
    parser.add_argument("--out-dir", default="results/pilot_pinch")
    args = parser.parse_args()

    reward_ablation.OUT_DIR = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    for seed in args.seeds:
        r = reward_ablation.run_one("pinch_v2", seed, args.timesteps, 8, args.reset_noise)
        last = r["curve"][-1]
        first = next((p["timesteps"] for p in r["curve"] if p["success_rate"] > 0), None)
        print(f"pinch_v2 seed={seed} noise={args.reset_noise} 訓練 {r['train_seconds']:.0f}s | 首次成功於 {first} 步 | "
              f"訓練末期成功率 {last['success_rate'] * 100:.1f}% | "
              f"評估 det={r['eval']['deterministic']} stoch={r['eval']['stochastic']}", flush=True)
