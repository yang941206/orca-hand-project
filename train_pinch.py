"""
訓練 ORCA 手完成「捏合」任務(拇指指尖碰食指指尖)。
用 SubprocVecEnv 平行跑多個環境加速訓練。

執行前記得:
1. conda activate orca
2. pinch_task.py 要跟這個檔案放在同一個資料夾
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit

from pinch_task import OrcaPinchTask


def make_env():
    def _init():
        env = OrcaPinchTask()
        env = TimeLimit(env, max_episode_steps=300)  # 捏合任務比較簡單,步數上限抓短一點
        return env
    return _init


if __name__ == "__main__":
    N_ENVS = 8  # 建議設成你 CPU 核心數,或核心數的一半;不確定就先用 4 或 8 試試

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./orca_tensorboard/",
    )

    # 200_000 是先試的量級,如果效果還不夠好,可以拉更高重新訓練
    model.learn(total_timesteps=200_000)

    model.save("orca_pinch_ppo")
    print("訓練完成,模型已存成 orca_pinch_ppo.zip")

    env.close()
