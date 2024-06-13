import gym

from DQN import DQN
from DDPG import DDPG
from SAC import SAC
from DDQN import DDQN

from replayBuffer import ReplayBuffer
from configs import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper


# Environment from Unity
# environment = "Crawler"
environment = "GridWorld"
# action_type = "vector"
action_type = "image"
unity_env = UnityEnvironment(f"../Environments/{environment}")
env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

obs_space = None
if isinstance(env.observation_space, gym.spaces.Tuple):
    # Iterate through each space in the tuple
    for space in env.observation_space:
        if isinstance(space, gym.spaces.Box) and len(space.shape) == 3:  # Check if it's a 3D Box (likely an image)
            obs_space = space.shape
            print("Image dimensions found:", obs_space)
            break
        elif isinstance(env.observation_space[0], gym.spaces.Box) and len(env.observation_space) == 1:
            obs_space = env.observation_space[0].shape[0]
            print("Vector observation space size:", obs_space)
        else:
            # If no image space found, you can default to handling non-image spaces or raise an error
            print("No image observation space found.")
else:
    # This case handles a single observation space which is not a tuple
    if isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
        obs_space = env.observation_space.shape
        print("Image dimensions found:", obs_space)
    else:
        obs_space = env.observation_space.shape[0]
        print("Non-image observation space size:", obs_space)


# Handle discrete and continuous action spaces
if isinstance(env.action_space, gym.spaces.Discrete):
    action_space = env.action_space.n
elif isinstance(env.action_space, gym.spaces.Box):
    action_space = env.action_space.shape[0]
else:
    print("Error in action")
    action_space = None
print("Action space size:", action_space)

# Configs and Algorithm
alg = "DDQN"
# alg = "DDPG"
config = ddqn_config(obs_space, action_space, action_type)
# config = ddpg_config(obs_space, action_space, action_type)
agent = DDQN(config)
# agent = DDPG(config)
buffer = ReplayBuffer(config)

# Logger, TensorBarod
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = f'tensorboard/{alg}/data_{environment}_{current_time}'
train_writer = SummaryWriter(log_dir=log_dir)

# Training
steps = 0  # total number of steps
for i_episode in range(config['eps_len']):
    observation = env.reset()
    obs = observation[0]

    done = False
    truncated = False
    t = 0  # time steps within each episode
    ret = 0.  # episodic return

    while done is False and truncated is False:
        obs = torch.tensor(obs).to(config['device'])  # observe the environment state
        if config["obs_type"] == "image":
            obs = obs / 255.0

        action = agent.act_probabilistic(obs[None, :])

        # action = action.cpu().detach().numpy()[0, :]  # take action

        next_obs, reward, done, info = env.step(action)  # environment advance to next step
        next_obs = next_obs[0]

        if config["obs_type"] == "image":
            next_obs = next_obs.astype(np.float32) / 255.0

        buffer.append_memory(obs=obs,  # put the transition to memory
                             action=torch.from_numpy(np.array([action])),
                             reward=torch.from_numpy(np.array([reward])),
                             next_obs=torch.from_numpy(np.array(next_obs)),
                             done=done)

        loss = agent.update(buffer)  # agent learn

        if config["obs_type"] == "image":
            obs = next_obs * 255
        else:
            obs = next_obs

        t += 1
        steps += 1
        ret += reward  # update episodic return

        train_writer.add_scalar('Performance/episodic_return', ret, i_episode)  # plot
        train_writer.add_scalar('Performance/episodic_loss', loss, i_episode)  # plot
        agent.save_models(f'models/{alg}_{environment}_best_model.pth')

        if done or truncated:
            print("Episode {} finished after {} timesteps with reward {}".format(i_episode, t + 1, ret))
            break

env.close()
train_writer.close()

