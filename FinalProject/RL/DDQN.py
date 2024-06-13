import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import binomial, choice
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, dim_obs, dim_action, dims_hidden_neurons, obs_type):
        super(QNetwork, self).__init__()
        self.dim_action = dim_action
        self.obs_type = obs_type

        if obs_type == 'vector':
            n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
            layers = []
            for dim_in, dim_out in zip(n_neurons[:-1], n_neurons[1:]):
                layers.append(nn.Linear(dim_in, dim_out))
                layers.append(nn.ReLU())
            layers.pop()  # Remove the last ReLU for the output layer
            self.layers = nn.Sequential(*layers)
            self.output = nn.Linear(n_neurons[-2], n_neurons[-1])
        elif obs_type == 'image':
            self.layers = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 2, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3072, 512),
                nn.ReLU(),
                nn.Linear(512, dim_action),
            )

    def forward(self, x):
        return self.layers(x)


class DDQN:
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
        self.training_step = 0

        self.Q = QNetwork(config['dim_obs'], config['dim_action'], config['dims_hidden_neurons'],
                          config['obs_type']).to(self.device)
        self.Q_tar = QNetwork(config['dim_obs'], config['dim_action'], config['dims_hidden_neurons'],
                              config['obs_type']).to(self.device)
        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=config['lr'])

    def update(self, buffer):
        # sample from replay memory
        t = buffer.sample(self.batch_size)

        (s, a, r, sp, dones) = (t.obs, t.action, t.reward, t.next_obs, t.done)

        a = a.long()

        Q_current = self.Q(s).gather(1, a)

        # Select the action according to the main network, evaluate using the target network
        best_actions = self.Q(sp).argmax(dim=1, keepdim=True)
        Q_target = r + self.discount * self.Q_tar(sp).gather(1, best_actions) * (1 - dones.float())

        loss = F.mse_loss(Q_current, Q_target)

        self.optimizer_Q.zero_grad()
        loss.backward()
        self.optimizer_Q.step()

        if self.training_step % self.C == 0:
            self.Q_tar.load_state_dict(self.Q.state_dict())

        self.training_step += 1
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
            with torch.no_grad():
                Q = self.Q(observation)
            val, a = torch.max(Q, axis=1)
            a = a.item()
            self.Q.train()
        return a

    def act_deterministic(self, observation: torch.Tensor):
        self.Q.eval()
        with torch.no_grad():
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
