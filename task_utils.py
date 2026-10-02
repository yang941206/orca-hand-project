"""
任務共用的小工具:計算兩個 body 之間「碰撞表面」的最近距離。

為什麼不用 data.body(...).xpos?
xpos 是 body 座標原點(大約在關節的位置),不是指尖。實測捏合時兩指表面已經碰在一起,
兩個原點之間仍然差 1.76 公分,所以改用 MuJoCo 的 mj_geomDistance 直接算表面距離。
"""

import mujoco
import numpy as np


def collision_geoms(model: mujoco.MjModel, body_name: str) -> list[int]:
    """回傳某個 body 底下會參與碰撞的 geom id(contype != 0 的那些)。"""
    body_id = model.body(body_name).id
    return [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] == body_id and model.geom_contype[g] != 0
    ]


def fingertip_local_point(model: mujoco.MjModel, body_name: str, cap: float = 0.004) -> np.ndarray:
    """
    從碰撞 mesh 的頂點估計「指尖點」(body 座標)。

    末節手指的長度方向是 body 的 z 軸,取 z 最大的那 cap(預設 4 mm)範圍內所有頂點的中心。
    為什麼需要這個:只用「末節表面距離」的話,拇指碰到食指末節的側面也會被算成捏合
    (demo 實測:表面已接觸,但兩指尖點相距 42.6 mm)。
    """
    points = []
    for g in collision_geoms(model, body_name):
        mesh_id = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        points.append(model.mesh_vert[start:start + count] @ rot.reshape(3, 3).T + model.geom_pos[g])
    points = np.vstack(points)
    return points[points[:, 2] > points[:, 2].max() - cap].mean(axis=0)


def body_point_world(data: mujoco.MjData, body_name: str, local_point: np.ndarray) -> np.ndarray:
    """把 body 座標的點轉成世界座標。"""
    body = data.body(body_name)
    return body.xpos + body.xmat.reshape(3, 3) @ local_point


def surface_distance(model, data, geoms_a, geoms_b, max_dist: float = 0.2) -> float:
    """兩組 geom 之間的最近表面距離(公尺)。互相穿透時回傳 0,不給負值。"""
    fromto = np.zeros(6)
    dist = min(
        mujoco.mj_geomDistance(model, data, a, b, max_dist, fromto)
        for a in geoms_a for b in geoms_b
    )
    return max(float(dist), 0.0)


def apply_reset_noise(env, scale: float) -> None:
    """
    在 reset 後的初始關節角度加上均勻雜訊 U(-scale, scale)(單位 rad),並夾在關節可動範圍內。

    為什麼需要:orca_sim 的 reset 沒有任何隨機性,每回合都從完全相同的姿勢開始(實測 E04)。
    評估時策略又是 deterministic,「跑 N 回合」只會得到 N 次一模一樣的結果。
    加一點雜訊後,N 回合評估才看得出策略的穩健度。
    雜訊用 env.np_random 產生,所以 reset(seed=...) 給同一個 seed 會得到同一個初始姿勢(可重現)。
    """
    if scale <= 0:
        return
    model, data = env.model, env.data
    noise = env.np_random.uniform(-scale, scale, size=model.nq)
    qpos = data.qpos + noise
    limited = model.jnt_limited.astype(bool)
    low, high = model.jnt_range[:, 0], model.jnt_range[:, 1]
    qpos[limited] = np.clip(qpos[limited], low[limited], high[limited])
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
