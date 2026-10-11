"""
抓球任務:從桌上把網球拿起來(實驗紀錄 E28)。

場景:ball_lift_scene.py(手 + 升降機構 + 桌子 + 網球),可行性已由 E27 確認。

觀測(42 維):手部 17 個關節角度 + 17 個角速度、升降高度與速度(2)、球相對掌心的位置(3)、球的速度(3)
動作(18 維):17 個手指 / 手腕 ctrl + 1 個升降 ctrl(都是目標位置;訓練時再用 RescaleAction 正規化到 [-1, 1])

reward(每一步;全部 ≤ 0,只有成功時 +50):
  reach   = −(5 個指尖到球表面的平均距離,cm)
  contact = −0.5 × (5 − 碰到球的手指數)              ← 手指 = 拇指、食指、中指、無名指、小指
  lift    = 握住時:−10 × (1 − clip(球升高 / 6 cm, 0, 1));沒握住時:−10   ← 條件式啟動
  away    = −(球在水平方向離開初始位置的距離,cm)
  smooth  = −0.05 × ‖a_t − a_{t−1}‖²
  success = +50(同時結束回合)
「握住」= 至少 3 根手指碰到球,且其中包含拇指。
成功 = 握住且球升高 ≥ 4 cm,連續 5 步。

兩個刻意的設計(前一階段的教訓):
- 每步 reward 全部 ≤ 0:若每步是正分、成功又會結束回合,策略會學到「故意不成功」來多拿分。
- 球掉下桌不提前結束回合:在負分設計下,若「失敗」會結束回合,策略反而會故意把球弄掉來停止扣分。
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ball_lift_scene import HAND_JOINTS, LiftSceneConfig, build_lift_model, hand_qpos_indices
from task_utils import collision_geoms, surface_distance

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
TIP_BODIES = {"thumb": "right_thumb_dp", "index": "right_index_ip", "middle": "right_middle_ip",
              "ring": "right_ring_ip", "pinky": "right_pinky_ip"}


class OrcaBallLiftTask(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    FRAME_SKIP = 5
    LIFT_GOAL = 0.06            # reward 的「滿分」升高量
    SUCCESS_RISE = 0.04         # 成功門檻
    HOLD_STEPS = 5
    SUCCESS_BONUS = 50.0
    CONTACT_COEF = 0.5
    LIFT_COEF = 10.0
    ACTION_RATE_COEF = 0.05

    def __init__(self, reset_noise: float = 0.05, ball_xy_noise: float = 0.005,
                 cfg: LiftSceneConfig | None = None, render_mode: str | None = None):
        self.cfg = cfg or LiftSceneConfig()
        self.model = build_lift_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.reset_noise = reset_noise
        self.ball_xy_noise = ball_xy_noise
        self.render_mode = render_mode
        m = self.model

        self._hq = hand_qpos_indices(m)
        hand_ids = [m.joint(n).id for n in HAND_JOINTS]
        self._hv = np.array([m.jnt_dofadr[j] for j in hand_ids])
        self._hand_range = m.jnt_range[hand_ids].copy()
        self._lift_q = m.jnt_qposadr[m.joint("lift_z").id]
        self._lift_v = m.jnt_dofadr[m.joint("lift_z").id]
        self._ball_q = m.jnt_qposadr[m.joint("ball_free").id]
        self._ball_v = m.jnt_dofadr[m.joint("ball_free").id]
        self._ball_body = m.body("ball").id
        self._ball_geom = [m.geom("ball_geom").id]
        self._palm = m.body("right_palm").id
        self._tip_geoms = {f: collision_geoms(m, b) for f, b in TIP_BODIES.items()}
        self._finger_of_body = {}
        for b in range(m.nbody):
            name = m.body(b).name
            for f in FINGERS:
                if name.startswith(f"right_{f}_"):
                    self._finger_of_body[b] = f

        lo, hi = m.actuator_ctrlrange[:, 0].astype(np.float32), m.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_low, self.action_high = lo, hi
        self.action_space = spaces.Box(low=lo, high=hi, dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(42,), dtype=np.float64)

        self._renderer = None
        self._viewer = None
        self._prev_action = None
        self._hold = 0
        self._ball_start = np.zeros(3)

    # ---------- 量測 ----------
    def ball_pos(self):
        return self.data.xpos[self._ball_body].copy()

    def ball_rise(self):
        return float(self.ball_pos()[2] - self._ball_start[2])

    def tip_distances(self):
        return {f: surface_distance(self.model, self.data, g, self._ball_geom) for f, g in self._tip_geoms.items()}

    def touching_fingers(self):
        """碰到球的手指(拇指、食指、中指、無名指、小指);掌心與底座不算手指。"""
        fingers = set()
        bg = self._ball_geom[0]
        for c in self.data.contact[:self.data.ncon]:
            if bg in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == bg else c.geom1
                f = self._finger_of_body.get(self.model.geom_bodyid[other])
                if f:
                    fingers.add(f)
        return fingers

    def _obs(self):
        d = self.data
        rel = self.ball_pos() - d.xpos[self._palm]
        return np.concatenate([
            d.qpos[self._hq], d.qvel[self._hv],
            [d.qpos[self._lift_q], d.qvel[self._lift_v]],
            rel, d.qvel[self._ball_v:self._ball_v + 3],
        ])

    # ---------- reward ----------
    def reward_terms(self, action_delta: float):
        tips = self.tip_distances()
        fingers = self.touching_fingers()
        grasped = len(fingers) >= 3 and "thumb" in fingers
        rise = self.ball_rise()
        away = float(np.linalg.norm(self.ball_pos()[:2] - self._ball_start[:2]))
        lift_frac = float(np.clip(rise / self.LIFT_GOAL, 0.0, 1.0))
        return {
            "reach": -float(np.mean(list(tips.values()))) * 100,
            "contact": -self.CONTACT_COEF * (5 - len(fingers)),
            "lift": -self.LIFT_COEF * (1 - lift_frac) if grasped else -self.LIFT_COEF,
            "away": -away * 100,
            "smooth": -self.ACTION_RATE_COEF * action_delta,
        }, grasped, rise, fingers

    # ---------- gym 介面 ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.reset_noise > 0:   # 只對 17 個手部關節加雜訊(不動升降與球)
            q = d.qpos[self._hq] + self.np_random.uniform(-self.reset_noise, self.reset_noise, size=17)
            d.qpos[self._hq] = np.clip(q, self._hand_range[:, 0], self._hand_range[:, 1])
        if self.ball_xy_noise > 0:
            d.qpos[self._ball_q:self._ball_q + 2] += self.np_random.uniform(-self.ball_xy_noise, self.ball_xy_noise, size=2)
        mujoco.mj_forward(m, d)
        self._ball_start = self.ball_pos()
        self._prev_action = None
        self._hold = 0
        if self.render_mode == "human":
            self.render()
        return self._obs(), {"is_success": False}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_low, self.action_high)
        delta = 0.0 if self._prev_action is None else float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action.copy()
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.FRAME_SKIP)

        terms, grasped, rise, fingers = self.reward_terms(delta)
        ok = grasped and rise >= self.SUCCESS_RISE
        self._hold = self._hold + 1 if ok else 0
        success = self._hold >= self.HOLD_STEPS
        reward = sum(terms.values()) + (self.SUCCESS_BONUS if success else 0.0)
        info = {"is_success": success, "grasped": grasped, "rise": rise, "fingers": len(fingers), **terms}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), success, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
