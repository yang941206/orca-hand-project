from orca_sim import OrcaHandRight

env = OrcaHandRight()
obs, info = env.reset(seed=0)

print("observation_space:", env.observation_space)
print("action_space:", env.action_space)
print("obs shape:", obs.shape)

action = env.action_space.sample()
obs2, reward, terminated, truncated, info = env.step(action)
print("reward(預期一定是 0.0):", reward)

env.close()