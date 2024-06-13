# from gym_unity.envs import UnityEnv

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

env_id = "Environments/Basic"
unity_env = UnityEnvironment(env_id)
env = UnityToGymWrapper(unity_env, uint8_visual=True)
# env = UnityEnv(env_id, worker_id=2, use_visual=False, no_graphics=False)

