"""
把 E27 找到的 5 組指令渲染出來:握球後(1.5 秒)與抬起後(3.5 秒)兩個時間點。輸出 demo/lift_feasibility.png。
用法:python analysis/render_lift_feasibility.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import feasibility_lift as F  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]


def render(sim, az, size=340):
    r = mujoco.Renderer(sim.m, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = sim.cfg.ball.ball_pos + np.array([0, -0.03, 0.03])
    cam.distance, cam.azimuth, cam.elevation = 0.36, az, -12
    r.update_scene(sim.d, cam)
    img = r.render()
    r.close()
    return img


if __name__ == "__main__":
    sols = json.load(open(os.path.join(ROOT, "demo", "lift_feasibility.json"), encoding="utf-8"))
    sim = F.LiftSim()
    m, d = sim.m, sim.d
    fig, axes = plt.subplots(2, len(sols), figsize=(3.3 * len(sols), 7))
    for k, s in enumerate(sols):
        ctrl = np.array(s["ctrl"])
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        series = [0.0] * F.GRASP + list(np.linspace(0, F.LIFT_TARGET, F.RAISE)) + [F.LIFT_TARGET] * F.HOLD
        for t, lift in enumerate(series):
            d.ctrl[:17] = ctrl
            d.ctrl[17] = lift
            mujoco.mj_step(m, d, nstep=F.FRAME_SKIP)
            if t == F.GRASP - 1:
                axes[0, k].imshow(render(sim, 0))
        axes[1, k].imshow(render(sim, 0))
        rise = s["result"]["rise"] * 1000
        axes[0, k].set_title(f"seed {s['seed']}:握球後(1.5 s)", fontsize=10)
        axes[1, k].set_title(f"抬起後(3.5 s),球升高 {rise:.1f} mm", fontsize=10)
        for row in range(2):
            axes[row, k].axis("off")
    plt.tight_layout()
    out = os.path.join(ROOT, "demo", "lift_feasibility.png")
    plt.savefig(out, dpi=90)
    print("已存:", os.path.relpath(out, ROOT))
