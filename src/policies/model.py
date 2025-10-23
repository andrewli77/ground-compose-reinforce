import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from .policy_network import PolicyNetwork


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class GridACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.actor = PolicyNetwork(32*8*8 + obs_space["rm_state"].shape[0], action_space, hiddens=[128, 64], layer_norm=True, activation=nn.ReLU())

        self.critic = nn.Sequential(
            nn.Linear(32*8*8 + obs_space["rm_state"].shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.apply(init_params)
        self.actor.out_layer.weight.data *= 0.01

    def forward(self, obs):
        image_encoding = self.conv_net(obs.observation.permute(0,3,1,2))
        dist = self.actor(torch.cat([image_encoding, obs.rm_state], dim=1))
        value = self.critic(torch.cat([image_encoding, obs.rm_state], dim=1)).squeeze(1)
        return dist, value

class DrawerACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.actor = PolicyNetwork(78 + obs_space["rm_state"].shape[0], action_space, hiddens=[512, 512, 512], activation=nn.ReLU())
        self.critic = nn.Sequential(
            nn.Linear(78 + obs_space["rm_state"].shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.apply(init_params)
        self.actor.out_layer.weight.data *= 0.01

    def forward(self, obs):
        dist = self.actor(torch.cat([obs.observation, obs.rm_state], dim=1))
        value = self.critic(torch.cat([obs.observation, obs.rm_state], dim=1)).squeeze(1)
        return dist, value
