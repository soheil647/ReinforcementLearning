import torch
import torch.nn as nn
from typing import Tuple

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)


class DDPG(nn.Module):
    def __init__(self, config):
        super(DDPG,self).__init__()
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

        # TO DO: Perform the updates for the actor and critic networks


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
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
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
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)


