import time
from orca_sim import OrcaHandRight

env = OrcaHandRight(render_mode="human")   # 加這個參數就會跳出視窗
obs, info = env.reset()

for _ in range(500):   # 跑 500 步,讓你有時間看畫面
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1.0 / env.metadata["render_fps"])   # 讓速度接近真實時間,不會一閃而過

    if terminated or truncated:
        obs, info = env.reset()

env.close()