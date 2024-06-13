from mlagents_envs.registry import default_registry

env = default_registry["Worm"].make()
env.reset()
for _ in range(500):
  env.step()
env.close()