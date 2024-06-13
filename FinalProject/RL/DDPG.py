import torch.nn as nn
from typing import Tuple
import torch
import numpy as np
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(ActorNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).float()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).float()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        a = torch.tanh(self.output(x))
        return a


class QCriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(QCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        n_neurons = (dim_obs + dim_action,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).float()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).float()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)


class DDPG(nn.Module):
    def __init__(self, config):
        super(DDPG, self).__init__()
        torch.manual_seed(config['seed'])

        self.lr_actor = config['lr_actor']  # learning rate
        self.lr_critic = config['lr_critic']
        self.smooth = config['smooth']  # smoothing coefficient for target net
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size
        self.sig = config['sig']  # exploration noise

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.device = config['device']

        self.actor = ActorNet(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q = QCriticNet(dim_obs=self.dim_obs,
                            dim_action=self.dim_action,
                            dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.actor_tar = ActorNet(dim_obs=self.dim_obs,
                                  dim_action=self.dim_action,
                                  dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q_tar = QCriticNet(dim_obs=self.dim_obs,
                                dim_action=self.dim_action,
                                dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr_critic)

    def update(self, buffer):
        # sample from replay memory
        t = buffer.sample(self.batch_size)

        (states, actions, rewards, next_states, dones) = (t.obs, t.action, t.reward, t.next_obs, t.done)

        # train q-network
        Q_current_value = self.Q(states, actions)
        Q_target_value = rewards + self.discount * self.Q_tar(next_states, self.actor_tar(next_states)) * (1 - dones.float())

        loss = F.mse_loss(Q_current_value, Q_target_value)

        self.optimizer_Q.zero_grad()
        loss.backward()
        self.optimizer_Q.step()

        # train actor-network
        greedy_actions = self.actor(states)
        loss = -self.Q(states, greedy_actions).mean()

        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()

        # Update target networks
        for target_param, param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.smooth * param.data + (1 - self.smooth) * target_param.data)

        for target_param, param in zip(self.actor_tar.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.smooth * param.data + (1 - self.smooth) * target_param.data)

        return loss.item()

    def act_probabilistic(self, obs: torch.Tensor):
        self.actor.eval()
        exploration_noise = torch.normal(torch.zeros(size=(self.dim_action,)), self.sig).to(self.device)
        a = self.actor(obs) + exploration_noise
        self.actor.train()
        return a

    def act_deterministic(self, obs: torch.Tensor):
        self.actor.eval()
        a = self.actor(obs)
        self.actor.train()
        return a

    def save_models(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.Q.state_dict(),
            'actor_target_state_dict': self.actor_tar.state_dict(),
            'critic_target_state_dict': self.Q_tar.state_dict(),
        }, filename)

    def load_models(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.Q.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_tar.load_state_dict(checkpoint['actor_target_state_dict'])
        self.Q_tar.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor.eval()
        self.Q.eval()
        self.actor_tar.eval()
        self.Q_tar.eval()

    def infer_action(self, obs: torch.Tensor):
        self.actor.eval()  # Ensure the network is in eval mode
        with torch.no_grad():  # Turn off gradients for inference, saves memory and computations
            action = self.actor(obs).squeeze(0).cpu().numpy()
        return action
