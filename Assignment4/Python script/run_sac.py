from sac import SAC
from buffer import ReplayBuffer
import os
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import datetime
import copy

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('Found device at: {}'.format(device))

env = gym.make('LunarLanderContinuous-v2',render_mode="human")

config = {
    'dim_obs': 8,
    'dim_action': 2,
    'dims_hidden_neurons': (120, 120),
    'lr': 0.001,
    'smooth': 0.99,
    'discount': 0.99,
    'alpha': 0.2,
    'batch_size': 128,
    'replay_buffer_size': 20000,
    'seed': 1,
    'max_episode': 500,
    'device':device
}

sac = SAC(config).to(device)
buffer = ReplayBuffer(config)
train_writer = SummaryWriter(log_dir='tensorboard/sac_{date:%Y-%m-%d_%H_%M_%S}'.format(
                             date=datetime.datetime.now()))

steps = 0
for i_episode in range(config['max_episode']):
    obs = env.reset()[0]
    done = False
    truncated = False
    t = 0
    ret = 0.
    while done is False and truncated is False:
        env.render()

        obs_tensor = torch.tensor(obs).type(Tensor).to(device)

        action = sac.act_probabilistic(obs_tensor[None, :]).detach().cpu().numpy()[0, :]
        next_obs, reward, done, truncated,_ = env.step(action)

        buffer.append_memory(obs=obs_tensor,
                             action=torch.from_numpy(action),
                             reward=torch.from_numpy(np.array([reward/10.0])),
                             next_obs=torch.from_numpy(next_obs).type(Tensor),
                             done=done)

        sac.update(buffer)

        t += 1
        steps += 1
        ret += reward

        obs = copy.deepcopy(next_obs)

        if done or truncated:
            print("Episode {} return {}".format(i_episode, ret))
    train_writer.add_scalar('Performance/episodic_return', ret, i_episode)

env.close()
train_writer.close()


