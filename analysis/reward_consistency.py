"""
重現 E08 / E09:握拳 reward 與「物理上真的握拳成功」的一致性分析(靜態分析)。

原本是在終端機臨時執行、沒有存檔的分析,這裡補成可重現的腳本(設定、亂數種子與當時相同)。
環境一律 reset_noise=0(這些分析做在加入初始姿勢雜訊之前)。

三種 reward(差別只在距離項):
- pose          :-加權姿勢誤差
- pose_tip      :-加權姿勢誤差 - 四指尖到掌心平均距離(公分)          ← 正式版
- pose_tip_thumb:-加權姿勢誤差 - 四指尖距離 - 拇指到食/中指距離(公分)

分析:
E08-1 ctrl = 標準姿勢 / v1 舊目標 的姿勢誤差與是否成功
E08-2 在標準姿勢附近隨機擾動 200 組 → 依姿勢誤差分組的成功率
E08-3 擾動 400 組中姿勢誤差 < 2 的,失敗是卡在哪個條件
E09-1 同一批 400 組擾動,三種 reward 的 AUC 與「reward 前 25 名的成功率」
E09-2 用 CEM 從手張開開始「只追 reward」(seed 1–6),最佳點會不會真的成功

用法(在專案根目錄):
    python analysis/reward_consistency.py
約需 8–10 分鐘(E09-2 的 CEM 最花時間)。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo_reward_v2 import cem, run_until_done, settle  # noqa: E402
from fist_task import FIST_TARGET_QPOS  # noqa: E402
from fist_task_v2 import FIST_STANDARD_QPOS, OrcaFistTaskV2  # noqa: E402

PERTURB_SCALE = 0.35  # 擾動幅度:每個關節 N(0, 0.35) × 半個動作範圍


def tip_cm(env):
    return float(np.mean(list(env.tip_distances().values()))) * 100


def reward_terms(env):
    """回傳三種 reward 的「負的距離總和」(不含動作平滑與成功 bonus,靜態姿勢下這兩項不影響排序)。"""
    pe, tc, th = env.pose_error(), tip_cm(env), env.thumb_distance() * 100
    return {"pose": -pe, "pose_tip": -(pe + tc), "pose_tip_thumb": -(pe + tc + th)}


def auc(score, ok):
    s, f = score[ok], score[~ok]
    return (s[:, None] > f[None]).mean() + 0.5 * (s[:, None] == f[None]).mean()


def perturbations(env, n, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = env.action_low, env.action_high
    for _ in range(n):
        yield np.clip(FIST_STANDARD_QPOS + rng.normal(0, PERTURB_SCALE, 17) * (hi - lo) / 2, lo, hi).astype(np.float32)


def e08(env):
    print("== E08-1 ==")
    for name, ctrl in [("ctrl = 標準姿勢", FIST_STANDARD_QPOS), ("ctrl = v1 舊目標", FIST_TARGET_QPOS)]:
        ok, steps = run_until_done(env, ctrl.astype(np.float32))
        settle(env, ctrl.astype(np.float32), 150)
        print(f"{name}:姿勢誤差 {env.pose_error():.2f},物理成功 = {ok}" + (f"(第 {steps} 步)" if ok else ""))

    print("\n== E08-2:擾動 200 組,依姿勢誤差分組 ==")
    rows = []
    for c in perturbations(env, 200):
        ok, _ = run_until_done(env, c, 120)
        settle(env, c, 100)
        rows.append((env.pose_error(), ok))
    rows = np.array(rows)
    for a, b in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 99)]:
        m = (rows[:, 0] >= a) & (rows[:, 0] < b)
        if m.sum():
            print(f"姿勢誤差 [{a},{b}):{int(m.sum()):3d} 組,物理成功率 {rows[m, 1].mean() * 100:5.1f}%")

    print("\n== E08-3:擾動 400 組中誤差 < 2 的失敗原因 ==")
    counts = {"成功": 0, "只有四指沒碰到掌心": 0, "只有拇指沒壓到": 0, "兩者都沒": 0}
    n = 0
    for c in perturbations(env, 400):
        settle(env, c, 100)
        if env.pose_error() >= 2:
            continue
        n += 1
        tips_ok = all(v < env.TIP_SUCCESS_DIST for v in env.tip_distances().values())
        thumb_ok = env.thumb_distance() < env.THUMB_SUCCESS_DIST
        key = ("成功" if tips_ok and thumb_ok else "兩者都沒" if not (tips_ok or thumb_ok)
               else "只有四指沒碰到掌心" if not tips_ok else "只有拇指沒壓到")
        counts[key] += 1
    print(f"誤差 < 2 的共 {n} 組:{counts}")


def e09(env):
    print("\n== E09-1:同一批 400 組擾動,三種 reward 的一致性 ==")
    scores, oks = {k: [] for k in ["pose", "pose_tip", "pose_tip_thumb"]}, []
    for c in perturbations(env, 400):
        settle(env, c, 100)
        for k, v in reward_terms(env).items():
            scores[k].append(v)
        oks.append(env._is_fist_pose())
    ok = np.array(oks, dtype=bool)
    print(f"物理成功(靜態判定)的姿勢:{ok.sum()} / 400")
    for k, s in scores.items():
        s = np.array(s)
        top25 = ok[np.argsort(-s)[:25]].mean() * 100
        print(f"{k:15s} AUC {auc(s, ok):.3f};reward 前 25 名的成功率 {top25:.0f}%")

    print("\n== E09-2:CEM 從手張開開始只追 reward(seed 1–6)==")
    lo, hi = env.action_low, env.action_high
    start = np.clip(env.reset()[0][:17], lo, hi)
    for k in scores:
        n_ok = 0
        for seed in range(1, 7):
            def score(x, k=k):
                settle(env, x)
                return -reward_terms(env)[k]
            x = cem(env, score, start, 0.5 * (hi - lo), seed=seed)
            ok_, _ = run_until_done(env, x)
            n_ok += ok_
            settle(env, x)
            print(f"  {k:15s} seed {seed}:成功 = {ok_!s:5}  指尖(mm) {[round(v * 1000) for v in env.tip_distances().values()]}"
                  f"  拇指 {env.thumb_distance() * 1000:.1f} mm")
        print(f"→ {k}:{n_ok}/6 成功")


if __name__ == "__main__":
    env = OrcaFistTaskV2(reset_noise=0.0)
    e08(env)
    e09(env)
