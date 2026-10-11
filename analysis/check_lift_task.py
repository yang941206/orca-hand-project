"""
檢查抓球 RL 環境 lift_task.py(實驗紀錄 E28)。

1. SB3 的 check_env
2. reset 的雜訊只影響手部關節(升降停在 0、球的高度不變、球 xy 偏移 ≤ 5 mm)
3. 5 種寫好的策略各跑 20 回合(seed 0–19),比較回合總 reward、成功率、球最後升高多少:
   隨機 / 手張開不動 / 手張開只往上抬 / 只握不抬 / E27 的握球 + 抬起
   → 真的把球拿起來的策略必須 reward 最高、而且是唯一會成功的,否則 reward 有漏洞
4. 「握球 + 抬起」與「只握不抬」各項 reward 的平均,看是哪一項拉開差距
5. 模擬速度

用法:python analysis/check_lift_task.py
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stable_baselines3.common.env_checker import check_env  # noqa: E402

from lift_task import OrcaBallLiftTask  # noqa: E402

EP_LEN = 300
TERMS = ["reach", "contact", "lift", "away", "smooth"]


def run_policy(env, policy, n=20):
    rets, succ, rises, steps_ok, term_sums = [], [], [], [], {k: [] for k in TERMS}
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        ret, ts = 0.0, {k: 0.0 for k in TERMS}
        for t in range(EP_LEN):
            obs, r, term, trunc, info = env.step(policy(env, t))
            ret += r
            for k in TERMS:
                ts[k] += info[k]
            if term:
                break
        rets.append(ret)
        succ.append(info["is_success"])
        rises.append(info["rise"])
        if info["is_success"]:
            steps_ok.append(t + 1)
        for k in TERMS:
            term_sums[k].append(ts[k] / (t + 1))
    return {"return": np.mean(rets), "success": np.mean(succ), "rise_mm": np.mean(rises) * 1000,
            "steps": np.mean(steps_ok) if steps_ok else None, "terms": {k: np.mean(v) for k, v in term_sums.items()}}


if __name__ == "__main__":
    env = OrcaBallLiftTask()
    check_env(env, warn=True)
    print("1. check_env 通過")

    env.reset(seed=3)
    d = env.data
    print(f"2. reset 後:升降 = {d.qpos[env._lift_q]:.4f};球高度變化 {(env.ball_pos()[2] - env.cfg.ball.ball_pos[2]) * 1000:.2f} mm;"
          f"球 xy 偏移 {np.abs(env.ball_pos()[:2] - env.cfg.ball.ball_pos[:2]).max() * 1000:.2f} mm;觀測維度 {env.observation_space.shape}")

    sol = {s["seed"]: s for s in json.load(open(os.path.join(ROOT, "demo", "lift_feasibility.json"), encoding="utf-8"))}
    grasp = np.array(sol[4]["ctrl"], dtype=np.float32)
    env.reset(seed=0)
    open_hand = np.clip(env.data.qpos[env._hq], env.action_low[:17], env.action_high[:17]).astype(np.float32)

    def lift_schedule(t):    # 與 E27 相同:握 1.5 秒 → 1 秒升到 6 cm → 維持
        return 0.0 if t < 150 else min(0.06, (t - 150) / 100 * 0.06)

    policies = {
        "隨機亂動": lambda e, t: e.action_space.sample(),
        "手張開不動": lambda e, t: np.append(open_hand, 0.0).astype(np.float32),
        "手張開、只往上抬": lambda e, t: np.append(open_hand, lift_schedule(t)).astype(np.float32),
        "只握住、不抬": lambda e, t: np.append(grasp, 0.0).astype(np.float32),
        "握球 + 抬起(E27 seed 4)": lambda e, t: np.append(grasp, lift_schedule(t)).astype(np.float32),
    }
    env.action_space.seed(0)
    results = {}
    for name, pol in policies.items():
        results[name] = r = run_policy(env, pol)
        print(f"3. {name:22s} 回合總 reward {r['return']:9.1f} | 成功率 {r['success'] * 100:5.1f}% | "
              f"球最後升高 {r['rise_mm']:6.1f} mm" + (f" | 平均 {r['steps']:.0f} 步成功" if r["steps"] else ""))

    for name in ["只握住、不抬", "握球 + 抬起(E27 seed 4)"]:
        t = results[name]["terms"]
        print(f"4. {name} 每步平均:" + "、".join(f"{k} {t[k]:.2f}" for k in TERMS))

    env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(2000):
        env.step(env.action_space.sample())
    print(f"5. 模擬速度(隨機動作,單環境):{2000 / (time.perf_counter() - t0):,.0f} steps/s")
