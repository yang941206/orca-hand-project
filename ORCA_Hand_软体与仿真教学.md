# ORCA Hand 软体安装与仿真完整教学

> 涵盖:仿真环境搭建(不需要实体手)→ 实体手到货后的驱动与校准 → sim-to-real 部署思路。
> 官方网站:https://orcahand.com ｜ GitHub 组织:https://github.com/orcahand

---

## 0. 先搞懂四个套件在做什么

ORCA 的软体分成四个各自独立的 Python 套件,依需求安装即可:

| 套件 | 作用 |
|---|---|
| **orca_core** | 驱动**实体**手的底层控制(串口通讯、张紧、校准、关节空间控制) |
| **orca_sim** | **纯仿真**环境,基于 MuJoCo + Gymnasium API,不需要实体手就能跑 |
| **orca_arm** | 完整平台的 URDF/MJCF 描述(手 + 机械臂一起用时才需要) |
| **orca_teleop** | 遥操作 / 动作重定向(VR 手把、MediaPipe 等输入源转成手部动作) |

你现在手还没组装好,**从 orca_sim 开始就对了**,后面手到了再装 orca_core。

---

## 1. 基础环境准备

建议用 Python 3.11 + 虚拟环境,避免污染系统环境。

### 方法 A:用 uv(官方推荐,较快)

```bash
# 先安装 uv(若尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows 用 PowerShell:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 建立虚拟环境
uv venv orca --python 3.11
source orca/bin/activate        # macOS / Linux
# orca\Scripts\activate         # Windows
```

### 方法 B:用 conda

```bash
conda create -n orca python=3.11 -y
conda activate orca
```

---

## 2. 第一步:安装并跑起仿真(orca_sim)

这一步**完全不需要实体手**,可以先在这里练操作、写控制逻辑、之后再谈训练。

### 2.1 安装

```bash
# 用 uv
uv pip install orca_sim

# 或用一般 pip
python -m pip install orca_sim
```

如果想跟上最新开发进度(套件仍在快速迭代中):

```bash
git clone https://github.com/orcahand/orca_sim
cd orca_sim && uv pip install -e .
```

> ⚠️ 官方提醒 orca_sim 还在持续更新中,若需要稳定版本,建议固定用 PyPI 上的版本(`pip install orca_sim`),不要一直跟 main branch。

### 2.2 跑起第一个仿真环境

orca_sim 遵循 **Gymnasium API**,物理引擎用 **MuJoCo**。最小范例:

```python
from orca_sim import OrcaHandRight  # 也有 OrcaHandLeft, OrcaHandCombined(左右手同时)

env = OrcaHandRight()
obs, info = env.reset()

obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

env.close()
```

跑起来后你会得到一个 MuJoCo 里的虚拟 ORCA 手,状态(关节角度等)透过 `obs` 拿到,动作透过 `env.step(action)` 下达 —— 这跟一般 RL 环境的写法完全一致。

### 2.3 可视化 / 渲染

MuJoCo 本身支援 render 出画面或影片,方便你直接用眼睛确认动作对不对。实际渲染模式(`human` / `rgb_array` 等)请以 `orca_sim` GitHub 上最新的 `examples/` 资料夹为准,因为这块 API 还在变动。

---

## 3. 在仿真里"预训练":大方向

由于 orca_sim 走的是标准 Gymnasium API,理论上你可以直接接上任何 RL 框架(如 Stable-Baselines3、RLlib、CleanRL 等)来训练策略,流程大致是:

1. 定义任务的 reward function(例如:抓取成功、姿态误差等)
2. 用你选的 RL 演算法(PPO / SAC 等常见于此类任务)对着 `OrcaHandRight()` 这类环境训练
3. 训练完的 policy 存下来(权重档)

> 这部分官方论文中有展示"仿真训练 → 零样本迁移到实体手"的案例(例如转球任务),说明流程本身是可行的。但具体训练脚本、reward 设计细节会一直更新,**建议直接去 GitHub `orcahand/orca_sim` 和相关论文附带的仓库看最新的 `examples/` 或训练脚本**,这里就不写死过时的代码。

---

## 4. 实体手到货后:orca_core 安装与校准

这段等你手组装好、接上电脑之后才需要。

### 4.1 取得代码并安装

```bash
git clone https://github.com/orcahand/orca_core
cd orca_core

# 建立环境并安装(含开发依赖)
uv sync --group dev
```

不用 uv 的话:

```bash
pip install .
```

### 4.2 确认硬体设定档

打开对应你手型号的 config 档,确认跟你实际组装的硬体一致,例如:

```
orca_core/models/v2/orcahand-right/config.yaml
```

档案里可选择手动指定(不写的话都会自动侦测):

```yaml
port: /dev/ttyACM0      # macOS 会是 /dev/cu.usbmodemXXXX,Windows 是 COM3 之类
baudrate: 1000000       # v2 用 1M;v1 用 3M
motor_type: dynamixel   # 或 feetech
```

### 4.3 张紧 → 校准 → 归零,三步跑完

```bash
uv run python scripts/tension.py orca_core/models/v2/orcahand-right/config.yaml
uv run python scripts/calibrate.py orca_core/models/v2/orcahand-right/config.yaml
uv run python scripts/neutral.py orca_core/models/v2/orcahand-right/config.yaml
```

(把路径换成你实际的手型资料夹)

### 4.4 常见环境问题

**Linux 串口权限**(出现 permission denied、抓不到马达):

```bash
sudo usermod -aG dialout $USER    # 永久生效,需重新登入
# 或暂时:
sudo chmod 666 /dev/ttyACM0
```

**Windows**:10/11 免额外设定权限,连接埠会显示成 `COM3`、`COM4`…,可用下面指令列出:

```bash
uv run python -m serial.tools.list_ports -v
```

---

## 5. Sim-to-real:把仿真练好的东西搬到实体手上

orca_sim(仿真)与 orca_core(实体)共用**同一套关节空间接口 / 状态格式**,这是整个设计的重点 —— 让你在仿真里训练好的 policy,理论上可以直接部署到实体手,不用重新训练(论文里也展示了这种零样本迁移的案例,像是转球任务)。

实务上通常流程是:

1. 在 orca_sim 里训练 / 验证 policy
2. 手到货、组装、跑完第 4 节的校准
3. 把仿真训练好的模型接到 orca_core 的控制介面上,读实体手的 `obs`、下 `action`
4. 可能需要做一些 domain randomization / 微调,缩小 sim-to-real 的落差

---

## 6. 进阶:遥操作与机械臂(选用)

- **orca_teleop**:让你用 VR 手把(如 Apple Vision Pro)、Rokoko 动作捕捉、或 MediaPipe(webcam 抓手部关键点)去远端操控手,也能拿来采集示范资料(demonstration data)做 imitation learning,支援与 lerobot 整合。
- **orca_arm**:如果你后续想把手装在机械臂上一起动,这个套件提供整个平台的 URDF/MJCF 描述档。

这两个目前文件相对少,进一步细节建议直接看对应 GitHub 仓库的 README。

---

## 7. 参考连结

- 官网:https://orcahand.com
- GitHub 组织(所有套件):https://github.com/orcahand
- orca_core:https://github.com/orcahand/orca_core
- orca_sim PyPI:https://pypi.org/project/orca-sim/
- 原始论文(ORCA hand 设计):arXiv:2504.04259
- 完整平台论文(orca_sim / orca_arm / orca_teleop 整合介绍):arXiv:2606.14561

---

## 建议你现在的顺序

1. ✅ 先装好 Python + uv/conda 环境
2. ✅ `pip install orca_sim`,跑通最小范例,确认仿真能动
3. ⏳ 熟悉 Gymnasium API、想清楚你要训练的任务(抓取?特定手势?)
4. ⏳ 等手组装完成,再回来做第 4 节的 orca_core 安装与校准
5. ⏳ 最后再谈 sim-to-real 部署
