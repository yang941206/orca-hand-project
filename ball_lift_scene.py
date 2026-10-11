"""
抓球任務(從桌上拿起來)的場景:ORCA 右手 + 升降機構 + 桌子 + 網球。

和 ball_scene.py 的差別(使用者決策,2026-10-12):要讓手「真的把球從桌上拿起來」,
所以在手的底座(right_tower)加一個上下滑動的關節和位置控制馬達,等於一隻只能上下動的簡化手臂。

設計依據:
- 球的位置用 E25 可行性檢查 seed 1 的結果(四指全部包覆、撤台後 2 秒只移動 0.1 mm):
  (x, y, z) = (0.0936, 0.0817, 0.1954),此時升降關節在 0(手的原始高度)。
- 桌面高度 = 球底(z − 半徑),球放在平坦的桌面上,不會像細支撐台那樣自己滾下來(E25)。
- 整隻手(含底座)重 1.117 kg → 升降馬達要撐住約 11 N;用位置控制,kp = 5000 N/m
  (重力造成的穩態誤差約 1.117 × 9.81 / 5000 ≈ 2.2 mm)。

注意:滑動關節掛在最上層的 right_tower,會變成模型的第一個關節,手指關節在 qpos 中的位置整體往後移一格。
使用這個場景的程式一律用關節名稱查位置(hand_qpos_indices()),不要寫死 qpos[:17]。
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ball_scene import BallSceneConfig, scene_xml_path

HAND_JOINTS = [
    "right_wrist",
    "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
    "right_index_abd", "right_index_mcp", "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd", "right_ring_mcp", "right_ring_pip",
    "right_pinky_abd", "right_pinky_mcp", "right_pinky_pip",
]


@dataclass
class LiftSceneConfig:
    ball: BallSceneConfig = field(default_factory=lambda: BallSceneConfig(
        ball_x=0.0936, gap=0.0817 - 0.024 - 0.0335, ball_z=0.1954, pedestal=False))
    lift_range: tuple = (-0.02, 0.10)      # 升降關節範圍(公尺);0 = 原始高度(球在可握住的位置)
    lift_kp: float = 5000.0                # 位置控制增益(N/m)
    lift_kv: float = 150.0                 # 速度阻尼(N·s/m)
    lift_force: float = 50.0               # 升降馬達最大出力(N)
    # 桌子的範圍:只需要蓋住球的正下方。拇指在 x≈0.03–0.05、z≈0.17,
    # 第一版桌子從 x≈0 開始,拇指指尖插進桌子 23.8 mm、手被卡住(E26)→ 桌子從 x = 0.065 開始。
    # 第二版桌子從 y = 0.040 開始,底座(right_tower,y 最大 +0.049、z 最高 0.159)插進桌子 9 mm
    # → 桌子從 y = 0.055 開始(球底接觸點 y = 0.0817,仍在桌面上)。
    table_x_range: tuple = (0.065, 0.220)
    table_y_range: tuple = (0.055, 0.200)
    table_friction: float = 1.0

    @property
    def table_top(self) -> float:
        return self.ball.ball_pos[2] - self.ball.ball_radius


def build_lift_model(cfg: LiftSceneConfig = LiftSceneConfig()) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(scene_xml_path())
    world = spec.worldbody

    # 1. 升降機構:在手的底座加一個沿 z 軸的滑動關節 + 位置控制馬達
    tower = spec.body("right_tower")
    tower.add_joint(
        name="lift_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
        range=list(cfg.lift_range), limited=True, damping=10.0, armature=0.01,
    )
    act = spec.add_actuator(name="lift_actuator", target="lift_z", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=cfg.lift_kp, kv=cfg.lift_kv)
    act.ctrllimited = True
    act.ctrlrange = list(cfg.lift_range)
    act.forcelimited = True
    act.forcerange = [-cfg.lift_force, cfg.lift_force]

    # 2. 桌子:從地板到桌面高度的實心方塊,放在手的正前方(+y)
    (x0, x1), (y0, y1) = cfg.table_x_range, cfg.table_y_range
    top = cfg.table_top
    world.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[(x1 - x0) / 2, (y1 - y0) / 2, top / 2],
        pos=[(x0 + x1) / 2, (y0 + y1) / 2, top / 2],
        friction=[cfg.table_friction, 0.005, 0.0001],
        rgba=[0.62, 0.52, 0.40, 1.0], contype=1, conaffinity=1, condim=3,
    )

    # 3. 球:放在桌面上
    b = cfg.ball
    ball = world.add_body(name="ball", pos=b.ball_pos)
    ball.add_freejoint(name="ball_free")
    ball.add_geom(
        name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[b.ball_radius, 0, 0],
        mass=b.ball_mass, friction=[b.ball_friction, 0.005, 0.0001],
        rgba=[0.80, 0.90, 0.20, 1.0], contype=1, conaffinity=1, condim=3,
    )
    return spec.compile()


def hand_qpos_indices(m: mujoco.MjModel) -> np.ndarray:
    """17 個手部關節在 qpos 中的位置(依 HAND_JOINTS 順序,與手的 actuator 順序相同)。"""
    return np.array([m.jnt_qposadr[m.joint(n).id] for n in HAND_JOINTS])
