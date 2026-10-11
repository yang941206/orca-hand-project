"""
檢查「從桌上拿球」場景(實驗紀錄 E26)。

1. 編譯成功、關節與馬達的數量和順序(升降關節變成第一個關節)
2. 手的 17 個馬達順序沒有被打亂(actuator 0–16 仍對應原本的手部關節)
3. 升降馬達能不能撐住手並到達指定高度(穩態誤差、多久穩定)
4. 升降範圍內,手(張開)與桌子的最近距離
5. 球放在桌上 2 秒是否穩定
6. 加了升降關節後,手指的動作是否跟原本一樣(同一組指令,比較手指關節角度)
7. 渲染 → demo/lift_scene.png

用法:python analysis/check_lift_scene.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ball_lift_scene import HAND_JOINTS, LiftSceneConfig, build_lift_model, hand_qpos_indices  # noqa: E402
from fist_task_v2 import FIST_STANDARD_QPOS  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
FRAME_SKIP = 5


def step(m, d, ctrl, n):
    for _ in range(n):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d, nstep=FRAME_SKIP)


def render(m, d, lookat, az, el=-15, dist=0.36, size=420):
    r = mujoco.Renderer(m, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    r.update_scene(d, cam)
    img = r.render()
    r.close()
    return img


if __name__ == "__main__":
    cfg = LiftSceneConfig()
    m = build_lift_model(cfg)
    d = mujoco.MjData(m)
    hq = hand_qpos_indices(m)
    lift_q = m.jnt_qposadr[m.joint("lift_z").id]
    lift_a = m.actuator("lift_actuator").id
    print(f"1. 編譯成功:nq={m.nq}(升降 1 + 手 17 + 球 7),nu={m.nu}(手 17 + 升降 1);"
          f"前 3 個關節 {[m.joint(j).name for j in range(3)]};升降關節 qpos 位置 {lift_q},升降馬達編號 {lift_a}")

    act_targets = [m.joint(m.actuator_trnid[a, 0]).name for a in range(17)]
    print(f"2. 手的馬達 0–16 對應的關節與 HAND_JOINTS 一致:{act_targets == HAND_JOINTS}")

    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    hand_open = np.clip(d.qpos[hq], lo[:17], hi[:17])
    print(f"   手的總質量(含底座){m.body_subtreemass[m.body('right_tower').id]:.3f} kg;桌面高度 {cfg.table_top:.4f} m;球心 {cfg.ball.ball_pos.round(4)}")

    for target in [0.0, 0.05, 0.08, 0.0]:
        ctrl = np.concatenate([hand_open, [target]])
        traj = []
        for k in range(100):
            step(m, d, ctrl, 1)
            traj.append(d.qpos[lift_q])
        err = np.array(traj) - target
        settle = next((k for k in range(100) if np.all(np.abs(err[k:]) < 0.002)), None)
        print(f"3. 升降目標 {target:.2f} m:1 秒後位置 {traj[-1]:.4f} m(誤差 {err[-1] * 1000:+.1f} mm);"
              f"進入 ±2 mm 的時間 {'未進入' if settle is None else f'{(settle + 1) * 0.01:.2f} s'}")

    ft = np.zeros(6)
    table = m.geom("table").id
    hand_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith("right_")]
    worst = (np.inf, None, None)
    for q in np.linspace(cfg.lift_range[0], cfg.lift_range[1], 13):
        mujoco.mj_resetData(m, d)
        d.qpos[lift_q] = q
        mujoco.mj_forward(m, d)
        dist, who = min((mujoco.mj_geomDistance(m, d, table, h, 0.3, ft), m.body(m.geom_bodyid[h]).name) for h in hand_geoms)
        if dist < worst[0]:
            worst = (dist, who, q)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    d0 = min(mujoco.mj_geomDistance(m, d, table, h, 0.3, ft) for h in hand_geoms)
    print(f"4a. 初始高度(0)時,張開的手與桌子最近距離 {d0 * 1000:.1f} mm")
    print(f"4b. 升降範圍內,張開的手與桌子最近距離 {worst[0] * 1000:.1f} mm(零件 {worst[1]},高度 {worst[2]:+.3f} m)")

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    b = m.body("ball").id
    p0 = d.xpos[b].copy()
    step(m, d, np.concatenate([hand_open, [0.0]]), 200)
    print(f"5. 球放在桌上、手不動 2 秒:球位移 {np.linalg.norm(d.xpos[b] - p0) * 1000:.2f} mm")

    from orca_sim import OrcaHandRight
    ref = OrcaHandRight()
    ref.reset()
    fist = np.clip(FIST_STANDARD_QPOS, lo[:17], hi[:17]).astype(np.float32)
    for _ in range(100):
        ref.step(fist)
    mujoco.mj_resetData(m, d)
    d.qpos[m.jnt_qposadr[m.joint("ball_free").id]:][:3] = [0.3, 0.3, 0.05]   # 把球移開,只比較手指
    mujoco.mj_forward(m, d)
    step(m, d, np.concatenate([fist, [0.0]]), 100)
    diff = np.abs(d.qpos[hq] - ref.data.qpos[:17]).max()
    print(f"6. 同一組手指指令 1 秒後,手指關節角度與原本環境的最大差異 {diff:.4f} rad")

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    look = cfg.ball.ball_pos + np.array([0, -0.03, 0.0])
    imgs = [(f"初始:方位角 {az}°", render(m, d, look, az)) for az in (270, 0)]
    step(m, d, np.concatenate([hand_open, [0.08]]), 100)
    imgs.append(("手升到 +0.08 m(方位角 0°)", render(m, d, look + [0, 0, 0.04], 0)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (t, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    out = os.path.join(ROOT, "demo", "lift_scene.png")
    plt.savefig(out, dpi=90)
    print("7. 已存:", os.path.relpath(out, ROOT))
