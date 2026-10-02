"""
從「物理上合法的握拳」樣本反推握拳標準姿勢(平均)與各關節容許誤差(標準差)。

流程:
1. 從「四指彎到底」附近加隨機擾動當起點,用 CEM 搜尋能滿足握拳物理條件
   (指尖到掌心 < 15 mm、拇指壓在食/中指上 < 10 mm,維持 5 步)的 ctrl
2. 只保留真的成功的樣本,讓手穩定後記錄 qpos
3. 平均 → FIST_STANDARD_QPOS;標準差 → FIST_JOINT_STD

結果寫在 fist_task_v2.py 的常數裡。這個腳本是為了讓數字可以重現,不用每次訓練都跑。

用法:
    python derive_fist_standard.py              # 有快取 demo/fist_samples.npz 就直接讀
    python derive_fist_standard.py --research   # 重新搜尋(8 個樣本約 4 分鐘)
"""

import argparse
import os

import numpy as np

from demo_reward_v2 import cem, run_until_done, settle
from fist_task_v2 import OrcaFistTaskV2

SAMPLES_PATH = "demo/fist_samples.npz"
FLEX_JOINTS = [6, 7, 9, 10, 12, 13, 15, 16]  # 四指 mcp / pip


def collect_samples(n_seeds: int = 8):
    env = OrcaFistTaskV2()
    lo, hi = env.action_low, env.action_high
    span = hi - lo

    def fist_score(x):
        settle(env, x)
        tips = env.tip_distances()
        thumb = env.thumb_distance()
        return max(max(tips.values()) - 0.006, 0) + max(thumb - 0.004, 0) + 0.1 * (sum(tips.values()) + thumb)

    base = env.data.qpos.copy()
    base[FLEX_JOINTS] = hi[FLEX_JOINTS]

    qpos_list, ctrl_list = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(200 + seed)
        mu = np.clip(base + rng.normal(0, 0.3, 17) * span * 0.5, lo, hi)
        ctrl = cem(env, fist_score, mu, 0.25 * span, seed=seed)
        ok, steps = run_until_done(env, ctrl)
        print(f"seed {seed}: 成功={ok}" + (f"(第 {steps} 步)" if ok else "(捨棄)"))
        if ok:
            settle(env, ctrl, 150)
            qpos_list.append(env.data.qpos.copy())
            ctrl_list.append(ctrl)

    qpos = np.array(qpos_list)
    os.makedirs("demo", exist_ok=True)
    np.savez(SAMPLES_PATH, qpos=qpos, ctrl=np.array(ctrl_list), mean=qpos.mean(0))
    return qpos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--research", action="store_true")
    args = parser.parse_args()

    if args.research or not os.path.exists(SAMPLES_PATH):
        qpos = collect_samples()
    else:
        qpos = np.load(SAMPLES_PATH)["qpos"]

    np.set_printoptions(precision=3, suppress=True)
    print(f"成功樣本數:{len(qpos)}")
    print("FIST_STANDARD_QPOS =", repr(qpos.mean(0).round(3)))
    print("FIST_JOINT_STD     =", repr(qpos.std(0).round(3)))
