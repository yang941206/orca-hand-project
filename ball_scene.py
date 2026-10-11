"""
抓球任務(方向 A)的場景:ORCA 右手 + 一顆球 + 支撐台。

做法:用 MuJoCo 的 MjSpec 讀進 orca_sim 原本的場景檔(scene_right.xml),在程式裡加上球和支撐台後
再編譯成模型,不修改 orca_sim 套件本身的檔案。

位置依據(實測,見 實驗紀錄 E24):
- 手固定在底座上、手指朝上(+z),重力朝 -z。
- 四指指根沿 x 軸排列(食指 x≈0.046 → 小指 x≈0.134),手指彎曲時指尖往 +y 移動
  → 掌心朝 +y,拇指也在 +y 側。掌心碰撞形狀的 y 最大值約 0.024。
- 所以球放在掌心正前方(+y)。最初放在 x=0.080、z=0.190 會穿過拇指第二節(thumb_pp)1.1 mm;
  改用網格搜尋(2,080 個候選位置)挑「與手所有零件距離約 5 mm、離掌心最近」的位置:x=0.095、z=0.180。

球的規格:網球(ORCA 原論文在真機上用網球做 RL 實驗)。
國際網球總會規範:直徑 6.54–6.86 cm、質量 56.0–59.4 g → 取半徑 3.35 cm、57 g。
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from orca_sim import OrcaHandRight


@dataclass
class BallSceneConfig:
    ball_radius: float = 0.0335            # 公尺,網球
    ball_mass: float = 0.057               # 公斤,網球
    ball_friction: float = 1.0             # 滑動摩擦係數(MuJoCo 預設值,與手的碰撞形狀相同)
    palm_front_y: float = 0.024            # 掌心碰撞形狀最前面的 y(實測)
    gap: float = 0.002                     # 球與掌心碰撞形狀最前端之間的空隙(y 方向)
    ball_x: float = 0.095                  # 中指與無名指之間、稍微遠離拇指(網格搜尋結果)
    ball_z: float = 0.180                  # 掌心中段(掌心 z 範圍約 0.138–0.223)
    pedestal: bool = True                  # 是否有支撐台
    pedestal_radius: float = 0.010         # 支撐台半徑(細一點,減少擋到手指的機會)
    ball_gravcomp: float = 0.0             # 球的重力補償(1 = 不受重力);可行性檢查用「虛擬支撐」時設 1。
                                           # 注意:必須在編譯時就不為 0,MuJoCo 才會計算重力補償(見 E25)

    @property
    def ball_pos(self) -> np.ndarray:
        return np.array([self.ball_x, self.palm_front_y + self.gap + self.ball_radius, self.ball_z])


def scene_xml_path() -> str:
    """orca_sim 原本的右手場景檔路徑(從環境物件取得,不寫死安裝位置)。"""
    env = OrcaHandRight()
    path = str(env.scene_path)
    env.close()
    return path


def build_ball_model(cfg: BallSceneConfig = BallSceneConfig()) -> mujoco.MjModel:
    """回傳加上球(和支撐台)之後編譯好的 MjModel。"""
    spec = mujoco.MjSpec.from_file(scene_xml_path())
    world = spec.worldbody

    ball = world.add_body(name="ball", pos=cfg.ball_pos, gravcomp=cfg.ball_gravcomp)
    ball.add_freejoint(name="ball_free")
    ball.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[cfg.ball_radius, 0, 0],
        mass=cfg.ball_mass,
        friction=[cfg.ball_friction, 0.005, 0.0001],
        rgba=[0.80, 0.90, 0.20, 1.0],      # 網球的黃綠色
        contype=1, conaffinity=1, condim=3,
    )

    if cfg.pedestal:
        height = cfg.ball_pos[2] - cfg.ball_radius   # 從地板撐到球底
        world.add_geom(
            name="pedestal",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[cfg.pedestal_radius, height / 2, 0],
            pos=[cfg.ball_pos[0], cfg.ball_pos[1], height / 2],
            rgba=[0.55, 0.55, 0.60, 1.0],
            contype=1, conaffinity=1, condim=3,
        )

    return spec.compile()
