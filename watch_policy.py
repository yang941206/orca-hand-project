"""
載入訓練好的模型,在 MuJoCo 視窗裡看它實際操控 ORCA 手完成握拳任務。
"""

import time
from stable_baselines3 import PPO

from fist_task import OrcaFistTask

env = OrcaFistTask(render_mode="human")
model = PPO.load("orca_fist_ppo")

obs, info = env.reset(seed=0)

for _ in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / env.metadata["render_fps"])

    if terminated or truncated:
        obs, info = env.reset()

env.close()
