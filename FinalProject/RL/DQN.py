import torch
import torch.nn as nn
from typing import Tuple, Text
from numpy.random import binomial
from numpy.random import choice
import torch.nn.functional as F
import numpy as np
from Layers import *


class QNetwork(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int],
                 obs_type: Text):
        if not isinstance(dim_obs, int):
            TypeError('dimension of observation must be int')
        if not isinstance(dim_action, int):
            TypeError('dimension of action must be int')
        if not isinstance(dims_hidden_neurons, tuple):
            TypeError('dimensions of hidden neurons must be tuple of int')

        super(QNetwork, self).__init__()
        self.num_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.obs_type = obs_type

        if self.obs_type == 'vector':
            n_neurons = (dim_obs, ) + dims_hidden_neurons + (dim_action, )
            for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
                layer = nn.Linear(dim_in, dim_out).double()
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                exec('self.layer{} = layer'.format(i + 1))

            self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
            torch.nn.init.xavier_uniform_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

        elif self.obs_type == 'image':
            self.conv_layers = nn.Sequential(
                Scale(1 / 255),
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 2, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3072, 512),
                nn.ReLU(),
                Linear0(512, self.dim_action),
            )

    def forward(self, observation: torch.Tensor):
        if self.obs_type == 'vector':
            x = observation.double()
            for i in range(self.num_layers):
                x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
            return self.output(x)
        elif self.obs_type == 'image':
            x = observation.float() / 255.0
            conv_out = self.conv_layers(x).view(x.size(0), -1)
            return conv_out
        else:
            return None


class DQN:
    def __init__(self, config):

        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.C = config['C']  # copy steps
        self.eps_len = config['eps_len']  # length of epsilon greedy exploration
        self.eps_max = config['eps_max']
        self.eps_min = config['eps_min']
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.obs_type = config['obs_type']

        self.device = config['device']

        self.Q = QNetwork(dim_obs=self.dim_obs,
                          dim_action=self.dim_action,
                          dims_hidden_neurons=self.dims_hidden_neurons,
                          obs_type=self.obs_type).to(self.device)
        self.Q_tar = QNetwork(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons,
                              obs_type=self.obs_type).to(self.device)

        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.training_step = 0

    def update(self, buffer):
        t = buffer.sample(self.batch_size)

        s = t.obs
        a = t.action
        r = t.reward
        sp = t.next_obs
        done = t.done

        self.training_step += 1

        a = a.long()
        # Calculate Q values for current states
        Q_current_value = self.Q(s).gather(1, a)

        # Compute target Q values
        Q_target_value = r + self.discount * self.Q_tar(sp).detach().max(1)[0].unsqueeze(1) * (1 - done.float())

        # # Calculate loss
        loss = F.mse_loss(Q_current_value, Q_target_value)

        # Optimize the Q network
        self.optimizer_Q.zero_grad()  # clear gradients since PyTorch accumulates them
        loss.backward()
        self.optimizer_Q.step()

        # # Update target Q network every C steps
        if self.training_step % self.C == 0:
            self.Q_tar.load_state_dict(self.Q.state_dict())

        return loss.item()

    def act_probabilistic(self, observation: torch.Tensor):
        # epsilon greedy:
        first_term = self.eps_max * (self.eps_len - self.training_step) / self.eps_len
        eps = max(first_term, self.eps_min)

        explore = binomial(1, eps)

        if explore == 1:
            a = choice(self.dim_action)
        else:
            self.Q.eval()
            Q = self.Q(observation)
            val, a = torch.max(Q, axis=1)
            a = a.item()
            self.Q.train()
        return a

    def act_deterministic(self, observation: torch.Tensor):
        self.Q.eval()
        Q = self.Q(observation)
        val, a = torch.max(Q, axis=1)
        self.Q.train()
        return a.item()

    def save_models(self, filename):
        torch.save({
            'critic_state_dict': self.Q.state_dict(),
            'critic_target_state_dict': self.Q_tar.state_dict(),
        }, filename)

    def load_models(self, filename):
        checkpoint = torch.load(filename)
        self.Q.load_state_dict(checkpoint['critic_state_dict'])
        self.Q_tar.load_state_dict(checkpoint['critic_target_state_dict'])
        self.Q.eval()
        self.Q_tar.eval()

    def infer_action(self, obs: torch.Tensor):
        self.Q.eval()  # Ensure the network is in eval mode
        with torch.no_grad():  # Turn off gradients for inference, saves memory and computations
            action_values = self.Q(obs)
            action = torch.argmax(action_values, dim=1).item()  # Select the action with the highest Q-value
        return action