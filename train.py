"""
訓練 ORCA 手完成「握拳」任務。
 
更正說明:orca_sim 目前這個版本並沒有內建的 OrcaHandRightCubeOrientation 任務環境
(之前的版本誤植了這個資訊)。OrcaHandRight 本身是一個「空白」環境,reward 永遠是 0、
沒有任何任務。握拳 reward 定義在 fist_task.py(繼承 OrcaHandRight),
細節、原理解釋請看 RL原理與自訂獎勵機制教學.md 的第 5、6 節。
"""
 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

from fist_task import OrcaFistTask  # 任務定義已抽到 fist_task.py


def make_env():
    env = OrcaFistTask()
    env = TimeLimit(env, max_episode_steps=500)   # 加上步數上限
    return env
 
 
if __name__ == "__main__":
    env = DummyVecEnv([make_env])
 
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./orca_tensorboard/",
    )
 
    # 先用小數字測試流程能不能跑通,確認沒問題後再拉高到 200_000 甚至更多
    model.learn(total_timesteps=500_000)
 
    model.save("orca_fist_ppo")
    print("訓練完成,模型已存成 orca_fist_ppo.zip")
 
    env.close()
 