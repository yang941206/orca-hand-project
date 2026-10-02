"""
載入訓練好的捏合任務模型,在 MuJoCo 視窗裡看它實際操控 ORCA 手。
"""

import time
from stable_baselines3 import PPO

from pinch_task import OrcaPinchTask

env = OrcaPinchTask(render_mode="human")
model = PPO.load("orca_pinch_ppo")

obs, info = env.reset(seed=0)

for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / env.metadata["render_fps"])

    if terminated or truncated:
        print("捏合成功!重新開始一輪")
        obs, info = env.reset()

env.close()
