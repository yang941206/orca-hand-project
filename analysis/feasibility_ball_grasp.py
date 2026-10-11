"""
抓球可行性檢查(實驗紀錄 E25):在不訓練 RL 的情況下,確認「撤掉支撐台後球留在手裡」物理上做不做得到。

方法:CEM(交叉熵法)同時搜尋 20 個數字
  - 球的位置 x、y(與掌心的間隙)、z
  - 17 個手指 / 手腕的 ctrl(固定不變的一組指令)
每組候選的測試流程:
  1. 手張開(預設姿勢),球以「虛擬支撐」固定在指定位置
  2. 送出 ctrl 1.5 秒(150 RL 步)讓手握住球
  3. 撤掉支撐(恢復重力),維持 ctrl 1 秒(100 步)
  4. 量球在撤台後移動多少

為什麼用「虛擬支撐」而不是實體支撐台(E25):
  - 在執行中修改實體支撐台的高度沒有生效(改高度後球直接穿過支撐台掉落;推測是 MuJoCo 編譯時建立的碰撞加速結構沒有更新)
  - 球放在 1 cm 粗的支撐台上本來就不穩(2 秒內慢慢滾動 17.9 mm)
  → 改用重力補償:握的階段球不受重力(body_gravcomp = 1)並加一點阻尼避免飄走;撤台 = 恢復重力、拿掉阻尼。
    這直接回答可行性的核心問題「手能不能抵抗重力握住球」。重力補償必須在編譯時就開啟(ngravcomp > 0)。
「握得住」的判定:撤台後球移動 < 1.5 cm,且結束時至少 3 個手部零件碰到球、其中包含拇指。
初始時球與手距離 < 2 mm 的候選直接淘汰(避免一開始就把球擠飛)。
注意:這裡的 ctrl 是「一組固定不變的指令」,做不到「先包住再收緊」這種有先後順序的動作,所以對 RL 來說偏悲觀。

用法(在專案根目錄):
    python analysis/feasibility_ball_grasp.py              # 預設 5 個 seed
    python analysis/feasibility_ball_grasp.py --seeds 0 1
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
from ball_scene import BallSceneConfig, build_ball_model  # noqa: E402

FRAME_SKIP = 5
GRASP_STEPS = 150          # 1.5 秒
HOLD_STEPS = 100           # 1 秒(最終驗證用 200 步 = 2 秒)
HOLD_TOL = 0.015           # 撤台後移動 < 1.5 cm 算握住
SUPPORT_DAMPING = 0.05     # 虛擬支撐時球的阻尼(避免被手指碰到後飄走)
MIN_CLEARANCE = 0.002      # 初始球與手至少 2 mm

# 球位置的搜尋範圍
X_RANGE = (0.060, 0.125)
GAP_RANGE = (0.002, 0.030)   # 球表面到掌心最前端(y 方向)
Z_RANGE = (0.165, 0.250)

CFG = BallSceneConfig(pedestal=False, ball_gravcomp=1.0)


class GraspSim:
    """包一個可重複使用的模型:每次評估只改球的位置與虛擬支撐的開關,不重新編譯。"""

    def __init__(self):
        self.m = build_ball_model(CFG)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.ball_body = m.body("ball").id
        self.ball_geom = m.geom("ball_geom").id
        self.qadr = m.jnt_qposadr[m.joint("ball_free").id]
        self.vadr = m.jnt_dofadr[m.joint("ball_free").id]
        self.hand_geoms = [g for g in range(m.ngeom)
                           if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith("right_")]
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self._ft = np.zeros(6)

    def set_support(self, on: bool):
        """虛擬支撐:on = 球不受重力並有阻尼;off = 恢復重力、拿掉阻尼(等於撤台)。"""
        self.m.body_gravcomp[self.ball_body] = 1.0 if on else 0.0
        self.m.dof_damping[self.vadr:self.vadr + 6] = SUPPORT_DAMPING if on else 0.0

    def reset(self, ball_pos):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr:self.qadr + 3] = ball_pos
        d.qpos[self.qadr + 3:self.qadr + 7] = [1, 0, 0, 0]
        self.set_support(True)
        mujoco.mj_forward(m, d)

    def clearance(self):
        return min(mujoco.mj_geomDistance(self.m, self.d, self.ball_geom, h, 0.2, self._ft) for h in self.hand_geoms)

    def run(self, ctrl, steps):
        for _ in range(steps):
            self.d.ctrl[:] = ctrl
            mujoco.mj_step(self.m, self.d, nstep=FRAME_SKIP)

    def ball_pos(self):
        return self.d.xpos[self.ball_body].copy()

    def touching(self):
        names = set()
        for c in self.d.contact[:self.d.ncon]:
            if self.ball_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.ball_geom else c.geom1
                n = self.m.body(self.m.geom_bodyid[other]).name
                if n.startswith("right_"):
                    names.add(n)
        return names

    def trial(self, ball_pos, ctrl, hold_steps=HOLD_STEPS):
        """回傳一次「握住 → 撤台 → 維持」的結果。"""
        self.reset(ball_pos)
        clear = self.clearance()
        if clear < MIN_CLEARANCE:
            return {"valid": False, "clearance": clear}
        self.run(ctrl, GRASP_STEPS)
        p_grasp = self.ball_pos()
        touch_grasp = self.touching()
        self.set_support(False)
        self.run(ctrl, hold_steps)
        p_end = self.ball_pos()
        touch_end = self.touching()
        moved = float(np.linalg.norm(p_end - p_grasp))
        thumb = any("thumb" in n for n in touch_end)
        return {
            "valid": True, "clearance": clear,
            "moved_after_removal": moved, "pushed_during_grasp": float(np.linalg.norm(p_grasp - ball_pos)),
            "touch_grasp": sorted(touch_grasp), "touch_end": sorted(touch_end),
            "held": moved < HOLD_TOL and len(touch_end) >= 3 and thumb,
        }


def decode(v):
    """把 20 維向量拆成 (球位置, ctrl)。前 3 維在 [0,1] 之間,對應位置範圍。"""
    u = np.clip(v[:3], 0, 1)
    x = X_RANGE[0] + u[0] * (X_RANGE[1] - X_RANGE[0])
    gap = GAP_RANGE[0] + u[1] * (GAP_RANGE[1] - GAP_RANGE[0])
    z = Z_RANGE[0] + u[2] * (Z_RANGE[1] - Z_RANGE[0])
    pos = np.array([x, CFG.palm_front_y + gap + CFG.ball_radius, z])
    return pos, v[3:]


def score(sim, v):
    """越小越好:撤台後的移動 + 0.3 × 握的過程中被推開的距離 + 接觸不足的懲罰。"""
    pos, ctrl = decode(v)
    r = sim.trial(pos, np.clip(ctrl, sim.lo, sim.hi))
    if not r["valid"]:
        return 1.0 + (MIN_CLEARANCE - r["clearance"]) * 10, r
    s = r["moved_after_removal"] + 0.3 * r["pushed_during_grasp"]
    s += 0.01 * max(0, 3 - len(r["touch_end"])) + (0 if any("thumb" in n for n in r["touch_end"]) else 0.01)
    return s, r


def cem(sim, seed, iters=40, pop=64, elite=10):
    rng = np.random.default_rng(seed)
    lo = np.concatenate([np.zeros(3), sim.lo])
    hi = np.concatenate([np.ones(3), sim.hi])
    mu = (lo + hi) / 2
    sigma = 0.5 * (hi - lo)
    best = (np.inf, None, None)
    for it in range(iters):
        X = np.clip(mu + sigma * rng.standard_normal((pop, len(mu))), lo, hi)
        results = [score(sim, x) for x in X]
        S = np.array([s for s, _ in results])
        idx = np.argsort(S)[:elite]
        if S[idx[0]] < best[0]:
            best = (S[idx[0]], X[idx[0]].copy(), results[idx[0]][1])
        mu, sigma = X[idx].mean(0), X[idx].std(0) + 1e-3
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    sim = GraspSim()

    # 0. 先確認虛擬支撐是有效的:不同位置撐 2 秒要不動,撤台後要掉
    for p in [CFG.ball_pos, np.array([0.10, CFG.palm_front_y + 0.02 + CFG.ball_radius, 0.23])]:
        sim.reset(p)
        sim.run(np.clip(sim.d.qpos[:17], sim.lo, sim.hi), 200)
        held = np.linalg.norm(sim.ball_pos() - p)
        sim.set_support(False)
        z1 = sim.ball_pos()[2]
        sim.run(np.clip(sim.d.qpos[:17], sim.lo, sim.hi), 50)
        print(f"0. 虛擬支撐檢查 {p.round(3)}:撐 2 秒移動 {held * 1000:.2f} mm;撤台後 0.5 秒下降 {(z1 - sim.ball_pos()[2]) * 1000:.0f} mm", flush=True)

    out = []
    for seed in args.seeds:
        t0 = time.time()
        s, v, r = cem(sim, seed)
        pos, ctrl = decode(v)
        ctrl = np.clip(ctrl, sim.lo, sim.hi)
        final = sim.trial(pos, ctrl, hold_steps=200)    # 最終驗證:撤台後維持 2 秒
        print(f"seed {seed}:{time.time() - t0:.0f}s | 球位置 {pos.round(4)} | "
              f"撤台後 2 秒移動 {final['moved_after_removal'] * 1000:.1f} mm | 握住={final['held']} | "
              f"握的過程推開 {final['pushed_during_grasp'] * 1000:.1f} mm | 結束時接觸 {final['touch_end']}", flush=True)
        out.append({"seed": seed, "ball_pos": pos.tolist(), "ctrl": ctrl.tolist(), "final": final})
    os.makedirs(os.path.join(ROOT, "demo"), exist_ok=True)
    with open(os.path.join(ROOT, "demo", "ball_feasibility.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    n = sum(o["final"]["held"] for o in out)
    print(f"結果:{n}/{len(out)} 個 seed 找到撤台後 2 秒仍握得住的(球位置, 指令)。存於 demo/ball_feasibility.json")
