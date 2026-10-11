"""
把 E25 可行性檢查找到的 5 組(球位置, 指令)渲染出來:撤台(恢復重力)2 秒後的狀態,兩個角度。
輸出 demo/ball_feasibility.png。用法:python analysis/render_feasibility.py
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
import feasibility_ball_grasp as F  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]


def render(sim, azimuth, elevation=-15, size=360):
    r = mujoco.Renderer(sim.m, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = sim.ball_pos() + np.array([0.0, -0.02, 0.02])
    cam.distance, cam.azimuth, cam.elevation = 0.30, azimuth, elevation
    r.update_scene(sim.d, cam)
    img = r.render()
    r.close()
    return img


if __name__ == "__main__":
    with open(os.path.join(ROOT, "demo", "ball_feasibility.json"), encoding="utf-8") as f:
        sols = json.load(f)
    sim = F.GraspSim()
    fig, axes = plt.subplots(2, len(sols), figsize=(3.4 * len(sols), 7.2))
    for k, s in enumerate(sols):
        r = sim.trial(np.array(s["ball_pos"]), np.array(s["ctrl"]), hold_steps=200)
        for row, az in enumerate((270, 0)):
            axes[row, k].imshow(render(sim, az))
            axes[row, k].axis("off")
        axes[0, k].set_title(f"seed {s['seed']}:撤台 2 秒後\n移動 {r['moved_after_removal'] * 1000:.1f} mm", fontsize=10)
    axes[0, 0].text(-0.05, 0.5, "掌心面", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.05, 0.5, "側面", transform=axes[1, 0].transAxes, rotation=90, va="center")
    plt.tight_layout()
    out = os.path.join(ROOT, "demo", "ball_feasibility.png")
    plt.savefig(out, dpi=90)
    print("已存:", os.path.relpath(out, ROOT))
