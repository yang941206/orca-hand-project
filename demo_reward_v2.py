"""
Demo:比較 v1(追姿勢 / body 原點距離)跟 v2(看結果 / 表面距離)兩種任務定義。

用法:
    python demo_reward_v2.py            # 產生比較圖 demo/compare_v1_v2.png
    python demo_reward_v2.py --view     # 開 MuJoCo 視窗依序播放三段示範

注意:這裡的動作不是 RL 學出來的,而是用最佳化(CEM)直接搜尋出的固定 ctrl,
目的只是證明「v2 的成功條件物理上達得到、v1 的達不到」。
demo/demo_poses.npz 不存在時會自動重新搜尋(約 2~3 分鐘)。
"""

import argparse
import os
import time

import matplotlib
import mujoco
import numpy as np

from fist_task import OrcaFistTask, FIST_TARGET_QPOS
from fist_task_v2 import OrcaFistTaskV2
from pinch_task import OrcaPinchTask
from pinch_task_v2 import OrcaPinchTaskV2

POSES_PATH = "demo/demo_poses.npz"
SETTLE_STEPS = 80


def settle(env, ctrl, n=SETTLE_STEPS):
    env.reset()
    for _ in range(n):
        env.step(ctrl)


def cem(env, score, mu, sigma, iters=30, pop=48, elite=8, seed=0):
    """簡單的 Cross-Entropy Method:反覆抽樣 ctrl、保留最好的幾組、更新分布。"""
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(iters):
        samples = np.clip(mu + sigma * rng.standard_normal((pop, len(mu))), env.action_low, env.action_high)
        scores = np.array([score(x) for x in samples])
        elite_idx = np.argsort(scores)[:elite]
        if scores[elite_idx[0]] < best[0]:
            best = (scores[elite_idx[0]], samples[elite_idx[0]].astype(np.float32))
        mu = samples[elite_idx].mean(0)
        sigma = samples[elite_idx].std(0) + 1e-3
    return best[1]


def search_poses():
    fist_env = OrcaFistTaskV2()

    def fist_score(x):
        settle(fist_env, x)
        tips = fist_env.tip_distances()
        thumb = fist_env.thumb_distance()
        return max(max(tips.values()) - 0.006, 0) + max(thumb - 0.004, 0) + 0.1 * (sum(tips.values()) + thumb)

    mu = fist_env.data.qpos.copy()
    for j in [6, 7, 9, 10, 12, 13, 15, 16]:  # 四指 mcp/pip 從最大彎曲開始搜
        mu[j] = fist_env.action_high[j]
    span = fist_env.action_high - fist_env.action_low
    fist_ctrl = cem(fist_env, fist_score, np.clip(mu, fist_env.action_low, fist_env.action_high), 0.25 * span)

    pinch_env = OrcaPinchTaskV2()
    lo, hi = pinch_env.action_low, pinch_env.action_high

    def side_score(x):  # 只要求末節表面碰到(舊的寬鬆定義),會找到「碰到側面」的姿勢
        settle(pinch_env, x)
        return pinch_env.contact_distance()

    def pinch_score(x):  # 指尖點靠近 + 表面接觸
        settle(pinch_env, x)
        return pinch_env.fingertip_distance() + pinch_env.contact_distance()

    side_ctrl = cem(pinch_env, side_score, (lo + hi) / 2, 0.5 * (hi - lo))
    pinch_ctrl = cem(pinch_env, pinch_score, (lo + hi) / 2, 0.5 * (hi - lo))

    os.makedirs("demo", exist_ok=True)
    np.savez(POSES_PATH, fist_ctrl=fist_ctrl, side_ctrl=side_ctrl, pinch_ctrl=pinch_ctrl)
    return fist_ctrl, side_ctrl, pinch_ctrl


def load_poses():
    if os.path.exists(POSES_PATH):
        data = np.load(POSES_PATH)
        return data["fist_ctrl"], data["side_ctrl"], data["pinch_ctrl"]
    print("找不到", POSES_PATH, ",重新用 CEM 搜尋…")
    return search_poses()


def run_until_done(env, ctrl, max_steps=300):
    """固定送同一組 ctrl,回傳 (是否成功, 第幾步結束)。"""
    env.reset()
    for t in range(1, max_steps + 1):
        _, _, terminated, _, _ = env.step(ctrl)
        if terminated:
            return True, t
    return False, max_steps


def build_scenes(fist_ctrl, side_ctrl, pinch_ctrl):
    """回傳三個示範場景,每個都附上 v1 / v2 的實測判定文字。"""
    scenes = []

    # 1. v1 握拳:ctrl 直接給目標姿態
    env = OrcaFistTask()
    ok, _ = run_until_done(env, FIST_TARGET_QPOS)
    settle(env, FIST_TARGET_QPOS)
    scenes.append({
        "title": "(a) v1 握拳:直接命令到 FIST_TARGET_QPOS",
        "env": env, "ctrl": FIST_TARGET_QPOS, "azimuth": 0,
        "text": f"qpos 距離 = {env._qpos_distance():.3f}(門檻 0.05)\n"
                f"拇指撞到食指/中指,被擋住\nv1 判定:{'成功' if ok else '失敗'}",
        "ok": ok,
    })

    # 2. v2 握拳
    env = OrcaFistTaskV2()
    ok, steps = run_until_done(env, fist_ctrl)
    settle(env, fist_ctrl)
    tips = env.tip_distances()
    scenes.append({
        "title": "(b) v2 握拳:指尖到掌心 + 拇指收起",
        "env": env, "ctrl": fist_ctrl, "azimuth": 0,
        "text": "指尖到掌心(mm):" + "、".join(f"{v * 1000:.0f}" for v in tips.values())
                + f"(門檻 {OrcaFistTaskV2.TIP_SUCCESS_DIST * 1000:.0f})\n"
                f"拇指到食/中指 = {env.thumb_distance() * 1000:.1f} mm(門檻 {OrcaFistTaskV2.THUMB_SUCCESS_DIST * 1000:.0f})\n"
                f"v2 判定:{'成功(第 ' + str(steps) + ' 步)' if ok else '失敗'}",
        "ok": ok,
    })

    # 3. 捏合(反例):只碰到食指側面
    env = OrcaPinchTaskV2()
    ok, _ = run_until_done(env, side_ctrl)
    settle(env, side_ctrl)
    scenes.append({
        "title": "(c) 捏合反例:拇指碰到食指「側面」",
        "env": env, "ctrl": side_ctrl, "azimuth": 0,
        "text": f"末節表面距離 = {env.contact_distance() * 1000:.1f} mm(有碰到)\n"
                f"但指尖點距離 = {env.fingertip_distance() * 1000:.1f} mm(門檻 10)\n"
                f"v2 判定:{'成功' if ok else '失敗(這不是捏合)'}",
        "ok": ok,
    })

    # 4. 捏合:指尖對指尖,v1 / v2 各自判定
    env_v1 = OrcaPinchTask()
    ok_v1, _ = run_until_done(env_v1, pinch_ctrl)
    settle(env_v1, pinch_ctrl)
    env = OrcaPinchTaskV2()
    ok_v2, steps = run_until_done(env, pinch_ctrl)
    settle(env, pinch_ctrl)
    scenes.append({
        "title": "(d) 捏合:指尖對指尖",
        "env": env, "ctrl": pinch_ctrl, "azimuth": 0,
        "text": f"v1 body 原點距離 = {env_v1._fingertip_distance() * 1000:.1f} mm(門檻 15)→ {'成功' if ok_v1 else '失敗'}\n"
                f"v2 指尖點 {env.fingertip_distance() * 1000:.1f} mm、表面 {env.contact_distance() * 1000:.1f} mm\n"
                f"v2 判定:{'成功(第 ' + str(steps) + ' 步)' if ok_v2 else '失敗'}",
        "ok": ok_v2,
    })
    return scenes


def render(env, azimuth, elevation=-20, distance=0.27, size=480):
    renderer = mujoco.Renderer(env.model, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = env.data.body("right_palm").xpos + np.array([0.0, 0.0, 0.04])
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    renderer.update_scene(env.data, cam)
    img = renderer.render()
    renderer.close()
    return img


def save_figure(scenes, path="demo/compare_v1_v2.png"):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, len(scenes), figsize=(5.2 * len(scenes), 11.2))
    for col, s in enumerate(scenes):
        for row, az in enumerate([s["azimuth"], 270]):
            ax = axes[row, col]
            ax.imshow(render(s["env"], az))
            ax.axis("off")
        axes[0, col].set_title(s["title"], fontsize=13)
        axes[1, col].text(0.5, -0.04, s["text"], transform=axes[1, col].transAxes, ha="center", va="top",
                          fontsize=11, color="#1a7f37" if s["ok"] else "#cf222e")
    axes[0, 0].text(-0.04, 0.5, "側面", transform=axes[0, 0].transAxes, rotation=90, va="center", fontsize=12)
    axes[1, 0].text(-0.04, 0.5, "掌心面", transform=axes[1, 0].transAxes, rotation=90, va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print("已存:", path)


def view(scenes):
    """開 MuJoCo 視窗,每個場景從張開的手慢慢動到目標 ctrl,停留幾秒。"""
    for s in scenes:
        env = type(s["env"])(render_mode="human")
        print("\n" + s["title"] + "\n" + s["text"])
        env.reset()
        for _ in range(240):  # 約 4 秒
            env.step(s["ctrl"])
            time.sleep(1.0 / env.metadata["render_fps"])
        time.sleep(2.0)
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="開 MuJoCo 視窗播放")
    parser.add_argument("--research", action="store_true", help="忽略快取,重新 CEM 搜尋")
    args = parser.parse_args()

    poses = search_poses() if args.research else load_poses()
    scenes = build_scenes(*poses)
    for s in scenes:
        print(s["title"], "|", s["text"].replace("\n", " | "))
    if args.view:
        view(scenes)
    else:
        save_figure(scenes)
