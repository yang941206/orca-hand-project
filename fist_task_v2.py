"""
握拳任務 v2(正式版):混合式設計。
- reward:追「從資料反推的標準握拳姿勢」,各關節依容許誤差加權
          + 四指指尖到掌心的平均距離(直接推手指把最後一段彎到底)
- 成功判定:看物理結果(指尖碰到掌心、拇指收進來),不是看關節角度

v1 的問題(實測):FIST_TARGET_QPOS 會讓拇指穿過食指/中指,有碰撞時物理上到不了,
最佳也只能到 qpos 距離 0.66(門檻 0.05),成功率永遠是 0。

標準姿勢怎麼來的(derive_fist_standard.py):
用最佳化找出 8 組物理上合法的握拳,統計各關節的平均值與標準差。
結果發現四指彎曲角度幾乎唯一(標準差 ≤ 0.05 rad),拇指跟左右張開角度則有很多種合法放法
(標準差 0.17~0.63 rad)。所以 reward 對四指彎曲要求嚴格、對拇指寬鬆,
避免把拇指綁死在某一種放法;最後算不算握拳,交給物理條件判定。

為什麼又加上指尖到掌心距離(實測):只用加權姿勢誤差時,在標準姿勢附近隨機擾動,
誤差 < 1 的姿勢只有 20% 真的握拳成功,失敗大多是四指沒彎到碰到掌心
(容許誤差下限 0.1 rad 對「最後一點點彎曲」太寬容)。加上指尖距離項,
讓 reward 直接反映物理成功條件中最常卡住的那一項。
曾經再加上「拇指到食/中指距離」項:靜態分析的一致性最好
(同一批 400 個擾動姿勢的 AUC:只有姿勢 0.894 → 加指尖 0.918 → 再加拇指 0.956),
但 PPO 實際訓練 100 萬步(reward_ablation.py,各 2 seed)結果相反:
  只有姿勢 87%/100%、加指尖 93%/99%、再加拇指 72%/21%(訓練末期成功率)
推測拇指項鼓勵拇指太早壓到食/中指上,擋住手指彎曲。所以正式版不用拇指項
(係數預設 0,保留參數讓 ablation 可以重現)。
"""

import numpy as np
from orca_sim import OrcaHandRight

from task_utils import collision_geoms, surface_distance


FINGERS = ["index", "middle", "ring", "pinky"]

# 關節順序:right_wrist, right_thumb_mcp/abd/pip/dip,
#          right_{index,middle,ring,pinky}_abd/mcp/pip
# 8 個合法握拳樣本的平均(derive_fist_standard.py 產生)。
# 實測 ctrl 直接給這組值,qpos 會穩定在距離 0.020 內,而且滿足握拳物理條件。
FIST_STANDARD_QPOS = np.array([
    0.039, 0.393, -0.404, 0.218, -0.250,
    -0.237, 1.620, 1.833,
    0.125, 1.576, 1.858,
    0.344, 1.579, 1.853,
    0.523, 1.587, 1.721,
], dtype=np.float64)

# 同 8 個樣本的標準差。0 代表每個樣本都彎到關節極限。
FIST_JOINT_STD = np.array([
    0.168, 0.282, 0.221, 0.333, 0.628,
    0.239, 0.024, 0.048,
    0.227, 0.005, 0.000,
    0.180, 0.000, 0.014,
    0.181, 0.032, 0.037,
], dtype=np.float64)

# 容許誤差下限 0.1 rad(約 6 度):避免標準差 0 的關節權重變成無限大,
# 而且只有 8 個樣本,標準差本身也不夠精確,不宜要求比這更嚴。
STD_FLOOR = 0.1
FIST_JOINT_WEIGHT = 1.0 / np.maximum(FIST_JOINT_STD, STD_FLOOR)


class OrcaFistTaskV2(OrcaHandRight):
    """繼承 OrcaHandRight,覆寫 reward、結束條件,並記錄上一步動作做平滑懲罰。"""

    # 每根指尖到掌心 < 1.5 公分。實測四指全彎時中指/無名指物理上最近只到約 0.9 公分
    # (碰不到掌心),所以門檻抓在這之上留一點餘裕
    TIP_SUCCESS_DIST = 0.015
    THUMB_SUCCESS_DIST = 0.010  # 拇指指尖到食/中指 < 1 公分
    HOLD_STEPS = 5              # 連續維持幾步才算成功
    SUCCESS_BONUS = 5.0
    ACTION_RATE_COEF = 0.05     # 動作變化量懲罰係數(讓動作平滑)
    TIP_DIST_COEF = 1.0         # 指尖到掌心平均距離(公分)的係數
    THUMB_DIST_COEF = 0.0       # 拇指到食/中指距離(公分)的係數;實測有害,正式版關掉

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.model
        self._palm = collision_geoms(m, "right_palm")
        self._tips = {f: collision_geoms(m, f"right_{f}_ip") for f in FINGERS}
        self._thumb_tip = collision_geoms(m, "right_thumb_dp")
        self._thumb_targets = (
            collision_geoms(m, "right_index_pp") + collision_geoms(m, "right_index_ip")
            + collision_geoms(m, "right_middle_pp") + collision_geoms(m, "right_middle_ip")
        )
        self._prev_action = None
        self._action_delta = 0.0
        self._hold_count = 0

    # ---- 量測 ----
    def pose_error(self) -> float:
        """加權關節誤差的 RMS,單位是「幾個容許誤差」。約 1 以下代表跟合法握拳樣本差不多。"""
        z = FIST_JOINT_WEIGHT * (self.data.qpos - FIST_STANDARD_QPOS)
        return float(np.sqrt(np.mean(z ** 2)))

    def tip_distances(self) -> dict[str, float]:
        return {f: surface_distance(self.model, self.data, g, self._palm) for f, g in self._tips.items()}

    def thumb_distance(self) -> float:
        return surface_distance(self.model, self.data, self._thumb_tip, self._thumb_targets)

    def _is_fist_pose(self, tips: dict[str, float] | None = None, thumb: float | None = None) -> bool:
        tips = self.tip_distances() if tips is None else tips
        thumb = self.thumb_distance() if thumb is None else thumb
        return all(d < self.TIP_SUCCESS_DIST for d in tips.values()) and thumb < self.THUMB_SUCCESS_DIST

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
        tips = self.tip_distances()  # 只算一次,reward 跟成功判定共用
        thumb = self.thumb_distance()
        tip_cm = float(np.mean(list(tips.values()))) * 100
        reward = (
            -self.pose_error()
            - self.TIP_DIST_COEF * tip_cm
            - self.THUMB_DIST_COEF * thumb * 100
            - self.ACTION_RATE_COEF * self._action_delta
        )

        self._hold_count = self._hold_count + 1 if self._is_fist_pose(tips, thumb) else 0
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
    env = OrcaFistTaskV2()
    obs, info = env.reset(seed=0)
    print("初始加權姿勢誤差:", env.pose_error())
    print("初始指尖到掌心(m):", env.tip_distances(), "拇指:", env.thumb_distance())

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward, "terminated:", terminated)

    env.close()
