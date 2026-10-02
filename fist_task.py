"""
自訂任務範例 1:握拳(讓所有關節角度靠近目標「握拳姿態」)。

原本寫在 train.py / watch_policy.py 裡,這裡抽出來獨立成模組,結構比照 pinch_task.py,
讓兩個任務可以共用同一套訓練/評估框架。

跟 pinch_task 的捏合範例差別在於:
- 握拳用的是「關節角度(qpos)距離目標姿態」
- 捏合用的是「兩個 body 的空間位置距離」
"""

import numpy as np
from orca_sim import OrcaHandRight


# 關節順序:right_wrist, right_thumb_mcp/abd/pip/dip,
#          right_{index,middle,ring,pinky}_abd/mcp/pip
# 數值都落在各關節實際可動範圍內(用 mujoco 載入模型確認過)。
FIST_TARGET_QPOS = np.array([
    0.0, 0.8, -0.3, 1.0, 1.2,
    0.0, 1.5, 1.7,
    0.0, 1.5, 1.7,
    0.0, 1.5, 1.7,
    0.0, 1.5, 1.7,
], dtype=np.float32)


class OrcaFistTask(OrcaHandRight):
    """繼承 OrcaHandRight,只覆寫 reward 跟結束條件。"""

    SUCCESS_DIST = 0.05  # 關節角度向量距離目標小於 0.05 rad 算握拳成功

    def _qpos_distance(self) -> float:
        current_qpos = self.data.qpos.copy()
        return float(np.linalg.norm(current_qpos - FIST_TARGET_QPOS))

    def _get_reward(self) -> float:
        return -self._qpos_distance()  # 越接近目標握拳姿態,reward 越高(越接近 0)

    def _get_terminated(self) -> bool:
        return self._qpos_distance() < self.SUCCESS_DIST


if __name__ == "__main__":
    # 簡單自我測試:確認 reward 會隨動作變化,且方向合理
    env = OrcaFistTask()
    obs, info = env.reset(seed=0)
    print("初始關節距離:", env._qpos_distance())

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward, "terminated:", terminated)

    env.close()
