"""
「從桌上拿球」可行性檢查(實驗紀錄 E27):固定的手指指令 + 預先寫好的升降動作,能不能把球從桌上拿起來?

流程(總共 3.5 秒 = 350 RL 步):
  1. 握球 1.5 秒:手指送固定 ctrl,升降停在 0(原始高度)
  2. 抬起 1 秒:升降目標從 0 線性增加到 +0.06 m,手指 ctrl 不變
  3. 維持 1 秒:升降停在 +0.06 m
「拿起來」的判定:結束時球比初始高 ≥ 4 cm,且至少 3 個手部零件碰到球、其中包含拇指。
CEM 只搜尋 17 個手指 ctrl(球與桌子位置固定)。
限制:手指 ctrl 從頭到尾不變,做不到「先張開、到位再收緊」→ 對 RL 來說偏悲觀。

用法:python analysis/feasibility_lift.py [--seeds 0 1 2 3 4]
"""

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ball_lift_scene import LiftSceneConfig, build_lift_model, hand_qpos_indices  # noqa: E402

FRAME_SKIP = 5
GRASP, RAISE, HOLD = 150, 100, 100
LIFT_TARGET = 0.06
SUCCESS_RISE = 0.04


class LiftSim:
    def __init__(self):
        self.cfg = LiftSceneConfig()
        self.m = build_lift_model(self.cfg)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.ball_body = m.body("ball").id
        self.ball_geom = m.geom("ball_geom").id
        self.hq = hand_qpos_indices(m)
        self.lo, self.hi = m.actuator_ctrlrange[:17, 0].copy(), m.actuator_ctrlrange[:17, 1].copy()

    def touching(self):
        names = set()
        for c in self.d.contact[:self.d.ncon]:
            if self.ball_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.ball_geom else c.geom1
                n = self.m.body(self.m.geom_bodyid[other]).name
                if n.startswith("right_"):
                    names.add(n)
        return names

    def trial(self, ctrl):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        p0 = d.xpos[self.ball_body].copy()
        lift_series = [0.0] * GRASP + list(np.linspace(0, LIFT_TARGET, RAISE)) + [LIFT_TARGET] * HOLD
        for k, lift in enumerate(lift_series):
            d.ctrl[:17] = ctrl
            d.ctrl[17] = lift
            mujoco.mj_step(m, d, nstep=FRAME_SKIP)
            if k == GRASP - 1:
                p_grasp = d.xpos[self.ball_body].copy()
                touch_grasp = self.touching()
        p_end = d.xpos[self.ball_body].copy()
        touch = self.touching()
        rise = float(p_end[2] - p0[2])
        thumb = any("thumb" in n for n in touch)
        return {
            "rise": rise, "pushed_during_grasp": float(np.linalg.norm(p_grasp - p0)),
            "touch_grasp": sorted(touch_grasp), "touch_end": sorted(touch),
            "lifted": rise >= SUCCESS_RISE and len(touch) >= 3 and thumb,
        }


def score(sim, ctrl):
    r = sim.trial(np.clip(ctrl, sim.lo, sim.hi))
    s = max(0.0, LIFT_TARGET - r["rise"]) + 0.3 * r["pushed_during_grasp"]
    s += 0.01 * max(0, 3 - len(r["touch_end"])) + (0 if any("thumb" in n for n in r["touch_end"]) else 0.01)
    return s, r


def cem(sim, seed, iters=40, pop=64, elite=10):
    rng = np.random.default_rng(seed)
    mu, sigma = (sim.lo + sim.hi) / 2, 0.5 * (sim.hi - sim.lo)
    best = (np.inf, None)
    for _ in range(iters):
        X = np.clip(mu + sigma * rng.standard_normal((pop, 17)), sim.lo, sim.hi)
        S = np.array([score(sim, x)[0] for x in X])
        idx = np.argsort(S)[:elite]
        if S[idx[0]] < best[0]:
            best = (S[idx[0]], X[idx[0]].copy())
        mu, sigma = X[idx].mean(0), X[idx].std(0) + 1e-3
    return best[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    sim = LiftSim()

    # 基準:手張開、只做升降,球應該留在桌上(確認不是升降本身把球帶起來)
    open_ctrl = np.clip(sim.d.qpos[sim.hq], sim.lo, sim.hi)
    mujoco.mj_resetData(sim.m, sim.d)
    mujoco.mj_forward(sim.m, sim.d)
    base = sim.trial(np.clip(sim.d.qpos[sim.hq], sim.lo, sim.hi))
    print(f"基準(手張開,只升降):球升高 {base['rise'] * 1000:.1f} mm,拿起來 = {base['lifted']}", flush=True)

    out = []
    for seed in args.seeds:
        t0 = time.time()
        ctrl = cem(sim, seed)
        r = sim.trial(ctrl)
        print(f"seed {seed}:{time.time() - t0:.0f}s | 球升高 {r['rise'] * 1000:.1f} mm | 拿起來 = {r['lifted']} | "
              f"握的過程推開 {r['pushed_during_grasp'] * 1000:.1f} mm | 結束時接觸 {r['touch_end']}", flush=True)
        out.append({"seed": seed, "ctrl": ctrl.tolist(), "result": r})
    with open(os.path.join(ROOT, "demo", "lift_feasibility.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"結果:{sum(o['result']['lifted'] for o in out)}/{len(out)} 個 seed 把球拿起來。存於 demo/lift_feasibility.json")
