"""
自訂任務範例 2:捏合(讓拇指指尖靠近食指指尖)。

跟 orca_fist_task 的握拳範例是同一套框架,差別在於:
- 握拳用的是「關節角度(qpos)距離目標姿態」
- 這裡用的是「兩個 body 的空間位置距離」,更適合「兩個部位要碰在一起」這種任務

body 名稱(right_thumb_dp / right_index_ip)已經用 mujoco 實際載入模型確認存在。
"""

import numpy as np
from orca_sim import OrcaHandRight


class OrcaPinchTask(OrcaHandRight):
    """繼承 OrcaHandRight,只覆寫 reward 跟結束條件。"""

    SUCCESS_DIST = 0.015  # 兩指尖距離小於 1.5 公分算捏合成功(可自行調整)

    def _fingertip_distance(self) -> float:
        thumb_tip = self.data.body("right_thumb_dp").xpos
        index_tip = self.data.body("right_index_ip").xpos
        return float(np.linalg.norm(thumb_tip - index_tip))

    def _get_reward(self) -> float:
        dist = self._fingertip_distance()
        reward = -dist  # 距離越小,reward 越高(越接近 0)

        # 額外加一個「完成 bonus」,鼓勵它不只是靠近,而是真的碰到
        if dist < self.SUCCESS_DIST:
            reward += 5.0

        return reward

    def _get_terminated(self) -> bool:
        return self._fingertip_distance() < self.SUCCESS_DIST


if __name__ == "__main__":
    # 簡單自我測試:確認 reward 會隨動作變化,且方向合理
    env = OrcaPinchTask()
    obs, info = env.reset(seed=0)
    print("初始指尖距離:", env._fingertip_distance())

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward, "terminated:", terminated)

    env.close()
