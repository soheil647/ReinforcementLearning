import gym

from stable_baselines3 import DQN
import matplotlib
matplotlib.use('Agg')

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import matplotlib.pyplot as plt

# from gym_unity.envs import UnityEnv
import numpy as np


def main():
    unity_env = UnityEnvironment("Environments/GridWorld")
    env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    done = False
    for j in range(10):
        cumulative_reward = 0
        steps = 0
        done = False
        while not done:
            random_action = env.action_space.sample()
            print(random_action)
            obs, reward, done, info = env.step(random_action)

            cumulative_reward += reward
            steps += 1
        print(cumulative_reward, steps)
        env.reset()
    env.close()


if __name__ == '__main__':
    main()