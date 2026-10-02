"""
捏合任務 v2:真正的「指尖對指尖」。

v1 的問題(實測):data.body(...).xpos 是 body 原點(靠近關節),不是指尖。
兩指表面已經碰在一起時,原點距離仍有 1.76 公分 > 門檻 1.5 公分,成功率永遠是 0。

只看「末節表面距離」也不夠(demo 實測):拇指碰到食指末節側面時表面距離 = 0,
但兩個指尖點其實相距 42.6 mm,那不是捏合。

v2 的定義:
- 指尖點:從碰撞 mesh 頂點算出的末節最前端中心(見 task_utils.fingertip_local_point)
- 成功:指尖點距離 < 10 mm,而且兩個末節表面確實接觸(< 2 mm),連續維持 HOLD_STEPS 步
  (實測 CEM 最佳可達指尖點距離 4.6 mm、表面接觸,所以門檻物理上達得到)
"""

import numpy as np
from orca_sim import OrcaHandRight

from task_utils import body_point_world, collision_geoms, fingertip_local_point, surface_distance


THUMB_BODY = "right_thumb_dp"
INDEX_BODY = "right_index_ip"


class OrcaPinchTaskV2(OrcaHandRight):
    """繼承 OrcaHandRight,覆寫 reward、結束條件,並記錄上一步動作做平滑懲罰。"""

    TIP_SUCCESS_DIST = 0.010      # 指尖點距離 < 10 mm
    CONTACT_DIST = 0.002          # 表面距離 < 2 mm 視為接觸
    HOLD_STEPS = 5
    SUCCESS_BONUS = 5.0
    ACTION_RATE_COEF = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.model
        self._thumb_geoms = collision_geoms(m, THUMB_BODY)
        self._index_geoms = collision_geoms(m, INDEX_BODY)
        self._thumb_tip_local = fingertip_local_point(m, THUMB_BODY)
        self._index_tip_local = fingertip_local_point(m, INDEX_BODY)
        self._prev_action = None
        self._action_delta = 0.0
        self._hold_count = 0

    # ---- 量測 ----
    def fingertip_distance(self) -> float:
        """兩個指尖點的直線距離(公尺)。"""
        thumb = body_point_world(self.data, THUMB_BODY, self._thumb_tip_local)
        index = body_point_world(self.data, INDEX_BODY, self._index_tip_local)
        return float(np.linalg.norm(thumb - index))

    def contact_distance(self) -> float:
        """兩個末節的碰撞表面最近距離(公尺)。"""
        return surface_distance(self.model, self.data, self._thumb_geoms, self._index_geoms)

    def _is_pinch(self) -> bool:
        return self.fingertip_distance() < self.TIP_SUCCESS_DIST and self.contact_distance() < self.CONTACT_DIST

    # ---- gym 介面 ----
    def reset(self, *, seed=None, options=None):
        self._prev_action = None
        self._action_delta = 0.0
        self._hold_count = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self._prev_action is not None:
            self._action_delta = float(np.sum((action - self._prev_action) ** 2))
        self._prev_action = action.copy()
        return super().step(action)

    def _get_reward(self) -> float:
        reward = -self.fingertip_distance() * 100 - self.ACTION_RATE_COEF * self._action_delta  # 公分為單位

        self._hold_count = self._hold_count + 1 if self._is_pinch() else 0
        if self._hold_count >= self.HOLD_STEPS:
            reward += self.SUCCESS_BONUS
        return float(reward)

    def _get_terminated(self) -> bool:
        # base class 的 step 會先呼叫 _get_reward 再呼叫這裡,所以 _hold_count 已經更新
        return self._hold_count >= self.HOLD_STEPS

    def _get_info(self) -> dict:
        return {"is_success": self._hold_count >= self.HOLD_STEPS}


if __name__ == "__main__":
    # 簡單自我測試:確認 reward 會隨動作變化,且方向合理
    env = OrcaPinchTaskV2()
    obs, info = env.reset(seed=0)
    print("初始指尖點距離(m):", env.fingertip_distance(), "表面距離(m):", env.contact_distance())

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward, "terminated:", terminated)

    env.close()
