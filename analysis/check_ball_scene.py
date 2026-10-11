"""
檢查抓球場景(方向 A)是否正確(實驗紀錄 E24)。

檢查項目:
1. 編譯成功;關節 / 自由度數量(手 17 + 球的自由關節)
2. 初始時球是否碰到手(不應該)
3. 有支撐台時球是否穩定不動;沒有支撐台時球是否會掉(確認重力與碰撞)
4. 加入球之後的模擬速度
5. 初步握球測試:ctrl 直接給「握拳標準姿勢」,記錄接觸到球的手指;
   再關掉支撐台的碰撞(模擬撤台),看球會不會掉 → 只是初步觀察,不是正式的可行性檢查
6. 渲染多角度圖片 → demo/ball_scene.png

用法(在專案根目錄):
    python analysis/check_ball_scene.py
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ball_scene import BallSceneConfig, build_ball_model  # noqa: E402
from fist_task_v2 import FIST_STANDARD_QPOS  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
FRAME_SKIP = 5            # 與 orca_sim 相同:每個 RL 步 = 5 次物理步 = 0.01 秒
HAND_BODIES = None


def hand_body_names(m):
    return {m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith("right_")}


def ball_contacts(m, d):
    """回傳與球接觸的手部 body 名稱集合。"""
    ball_geom = m.geom("ball_geom").id
    names = set()
    for c in d.contact[:d.ncon]:
        if ball_geom in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == ball_geom else c.geom1
            name = m.body(m.geom_bodyid[other]).name
            if name.startswith("right_"):
                names.add(name)
    return names


def run(m, d, ctrl, rl_steps):
    for _ in range(rl_steps):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d, nstep=FRAME_SKIP)


def default_ctrl(m, d):
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    return np.clip(d.qpos[:17], lo, hi)


def render(m, d, ball_pos, azimuth, elevation=-15, distance=0.32, size=420):
    r = mujoco.Renderer(m, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = ball_pos + np.array([0.0, -0.02, 0.02])
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    r.update_scene(d, cam)
    img = r.render()
    r.close()
    return img


if __name__ == "__main__":
    cfg = BallSceneConfig()
    m = build_ball_model(cfg)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    ball_id = m.body("ball").id
    print(f"1. 編譯成功:nq={m.nq}(手 17 + 球 7),nv={m.nv}(手 17 + 球 6),nu={m.nu},"
          f"球心 {cfg.ball_pos.round(4)},球半徑 {cfg.ball_radius} m,質量 {cfg.ball_mass} kg")

    ft = np.zeros(6)
    hand_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith("right_")]
    for name in ("ball_geom", "pedestal"):
        g = m.geom(name).id
        dist, who = min((mujoco.mj_geomDistance(m, d, g, h, 0.2, ft), m.body(m.geom_bodyid[h]).name) for h in hand_geoms)
        print(f"2. 初始時 {name} 與手的最近距離:{dist * 1000:.1f} mm(最近的零件 {who})")
    print(f"   初始時碰到球的手部零件:{sorted(ball_contacts(m, d)) or '無'}")

    start = d.xpos[ball_id].copy()
    run(m, d, default_ctrl(m, d), 200)   # 2 秒
    moved = np.linalg.norm(d.xpos[ball_id] - start)
    print(f"3a. 有支撐台、手不動、模擬 2 秒:球位移 {moved * 1000:.2f} mm;此時碰到球的手部零件 {sorted(ball_contacts(m, d)) or '無'}")

    m2 = build_ball_model(BallSceneConfig(pedestal=False))
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetData(m2, d2)
    mujoco.mj_forward(m2, d2)   # 先更新一次位置再讀(修正:原本讀到的是尚未計算的 0)
    z0 = d2.xpos[m2.body("ball").id][2]
    run(m2, d2, default_ctrl(m2, d2), 100)
    print(f"3b. 沒有支撐台、模擬 1 秒:球的高度 {z0:.3f} → {d2.xpos[m2.body('ball').id][2]:.3f} m(應該往下掉)")

    from orca_sim import OrcaHandRight
    base = OrcaHandRight()
    base.reset()
    t = time.perf_counter()
    for _ in range(2000):
        base.step(base.action_space.sample())
    v0 = 2000 / (time.perf_counter() - t)
    mujoco.mj_resetData(m, d)
    rng = np.random.default_rng(0)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    t = time.perf_counter()
    for _ in range(2000):
        d.ctrl[:] = rng.uniform(lo, hi)
        mujoco.mj_step(m, d, nstep=FRAME_SKIP)
    v1 = 2000 / (time.perf_counter() - t)
    print(f"4. 模擬速度(隨機動作,單環境):只有手 {v0:,.0f} steps/s;手 + 球 + 支撐台 {v1:,.0f} steps/s")

    # 5. 初步握球
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    fist = np.clip(FIST_STANDARD_QPOS, lo, hi)
    run(m, d, fist, 150)
    touching = sorted(ball_contacts(m, d))
    held_pos = d.xpos[ball_id].copy()
    print(f"5a. ctrl = 握拳標準姿勢 1.5 秒後:碰到球的手部零件 {touching};球位移 {np.linalg.norm(held_pos - cfg.ball_pos) * 1000:.1f} mm")
    img_grasp = [render(m, d, cfg.ball_pos, az) for az in (90, 180)]
    ped = m.geom("pedestal").id
    m.geom_contype[ped] = 0
    m.geom_conaffinity[ped] = 0          # 關掉支撐台的碰撞 = 撤台
    run(m, d, fist, 100)
    drop = held_pos[2] - d.xpos[ball_id][2]
    print(f"5b. 撤台後 1 秒:球下降 {drop * 1000:.1f} mm;碰到球的手部零件 {sorted(ball_contacts(m, d)) or '無'}")
    m.geom_contype[ped] = 1
    m.geom_conaffinity[ped] = 1

    # 6. 渲染初始場景
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    views = [(az, render(m, d, cfg.ball_pos, az)) for az in (0, 90, 180, 270)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for ax, (az, img) in zip(axes.flat[:4], views):
        ax.imshow(img)
        ax.set_title(f"初始場景(相機方位角 {az}°)")
        ax.axis("off")
    for ax, img, az in zip(axes.flat[4:], img_grasp, (90, 180)):
        ax.imshow(img)
        ax.set_title(f"握拳標準姿勢握球(方位角 {az}°)")
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(ROOT, "demo", "ball_scene.png")
    plt.savefig(out, dpi=90)
    print("6. 已存:", os.path.relpath(out, ROOT))
