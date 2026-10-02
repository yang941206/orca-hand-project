"""
捏合 v2 可學性試驗:PPO 預設超參數 100 萬步,確認 RL 學得會,再排正式實驗矩陣。
訓練/評估流程跟 reward_ablation.py 完全相同(共用它的函式),只換任務。

用法:
    python pilot_pinch.py
結果存到 results/pilot_pinch/
"""

import reward_ablation
from pinch_task_v2 import OrcaPinchTaskV2

reward_ablation.VARIANTS = {"pinch_v2": OrcaPinchTaskV2}
reward_ablation.OUT_DIR = "results/pilot_pinch"

if __name__ == "__main__":
    import os
    os.makedirs(reward_ablation.OUT_DIR, exist_ok=True)
    for seed in [0, 1]:
        r = reward_ablation.run_one("pinch_v2", seed, 1_000_000, 8)
        last = r["curve"][-1]
        first = next((p["timesteps"] for p in r["curve"] if p["success_rate"] > 0), None)
        print(f"pinch_v2 seed={seed} 訓練 {r['train_seconds']:.0f}s | 首次成功於 {first} 步 | "
              f"訓練末期成功率 {last['success_rate'] * 100:.1f}% | "
              f"評估 det={r['eval']['deterministic']} stoch={r['eval']['stochastic']}", flush=True)
