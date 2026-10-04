"""
重現 E04:驗證 v1 任務的成功條件在物理上達不到。

原本是在終端機臨時執行、沒有存檔的分析,這裡補成可重現的腳本(設定與當時相同)。

檢查項目:
1. 握拳:ctrl 直接給 FIST_TARGET_QPOS 跑 300 步,最終 qpos 距離(門檻 0.05)
2. 握拳:同上但關閉碰撞(mjDSBL_CONTACT),確認是不是碰撞擋住
3. 握拳:穩定後的接觸點(哪些 body 碰在一起)
4. 握拳:CEM 搜尋能達到的最小 qpos 距離
5. 捏合:隨機 300 組 ctrl 的最小 body 原點距離;CEM 最小距離;同姿勢下兩末節的表面距離(門檻 15 mm)
6. 既有的 v1 PPO 模型(orca_fist_ppo.zip / orca_pinch_ppo.zip)deterministic 跑一回合的結果
7. reset() 換 seed 時初始觀測是否不同

用法(在專案根目錄):
    python analysis/verify_v1_infeasible.py
約需 2–3 分鐘。
"""

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gymnasium.wrappers import TimeLimit  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from demo_reward_v2 import cem, settle  # noqa: E402
from fist_task import FIST_TARGET_QPOS, OrcaFistTask  # noqa: E402
from pinch_task import OrcaPinchTask  # noqa: E402
from task_utils import collision_geoms, surface_distance  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fist_checks():
    print("== 握拳 v1(門檻:qpos 距離 < 0.05)==")
    env = OrcaFistTask()
    env.reset(seed=0)
    for _ in range(300):
        env.step(FIST_TARGET_QPOS)
    print(f"1. ctrl = 目標姿勢 300 步後的 qpos 距離:{env._qpos_distance():.3f}")

    m, d = env.model, env.data
    pairs = sorted({tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)))
                    for c in d.contact[:d.ncon]})
    print("3. 穩定後互相接觸的 body:", "; ".join(f"{a}–{b}" for a, b in pairs))

    original_flags = m.opt.disableflags
    m.opt.disableflags = original_flags | mujoco.mjtDisableBit.mjDSBL_CONTACT
    env.reset(seed=0)
    for _ in range(300):
        env.step(FIST_TARGET_QPOS)
    print(f"2. 關閉碰撞後的 qpos 距離:{env._qpos_distance():.3f}")
    m.opt.disableflags = original_flags  # 恢復碰撞(mjtDisableBit 不支援 ~ 運算,所以存原值再還原)

    def score(x):
        settle(env, x)
        return env._qpos_distance()
    x = cem(env, score, FIST_TARGET_QPOS.astype(np.float64).copy(), 0.3 * np.ones(17), iters=25, seed=0)
    settle(env, x)
    print(f"4. CEM 能達到的最小 qpos 距離:{env._qpos_distance():.3f}")


def pinch_checks():
    print("\n== 捏合 v1(門檻:body 原點距離 < 15 mm)==")
    env = OrcaPinchTask()
    rng = np.random.default_rng(0)
    best = np.inf
    for _ in range(300):
        env.reset()
        act = rng.uniform(env.action_low, env.action_high)
        for _ in range(60):
            env.step(act)
        best = min(best, env._fingertip_distance())
    print(f"5a. 隨機 300 組 ctrl 的最小原點距離:{best * 1000:.1f} mm")

    def score(x):
        settle(env, x)
        return env._fingertip_distance()
    lo, hi = env.action_low, env.action_high
    x = cem(env, score, (lo + hi) / 2, 0.5 * (hi - lo), iters=30, seed=0)
    settle(env, x)
    m, d = env.model, env.data
    surf = surface_distance(m, d, collision_geoms(m, "right_thumb_dp"), collision_geoms(m, "right_index_ip"))
    print(f"5b. CEM 最小原點距離:{env._fingertip_distance() * 1000:.1f} mm;同姿勢下兩末節表面距離:{surf * 1000:.2f} mm"
          "(surface_distance 會把負值截成 0)")


def old_models():
    print("\n== 既有 v1 PPO 模型(deterministic,1 回合)==")
    for name, cls, steps in [("orca_fist_ppo", OrcaFistTask, 500), ("orca_pinch_ppo", OrcaPinchTask, 300)]:
        model = PPO.load(os.path.join(ROOT, name), device="cpu")
        env = TimeLimit(cls(), steps)
        obs, _ = env.reset(seed=0)
        total = 0.0
        for t in range(1, steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        print(f"6. {name}:{t} 步、回合總 reward {total:.2f}、成功 = {term}、最後一步 reward {r:.4f}")


def reset_randomness():
    a = OrcaPinchTask().reset(seed=0)[0]
    b = OrcaPinchTask().reset(seed=123)[0]
    print(f"\n7. reset(seed=0) 與 reset(seed=123) 的初始觀測完全相同:{np.allclose(a, b)}")


if __name__ == "__main__":
    fist_checks()
    pinch_checks()
    old_models()
    reset_randomness()
