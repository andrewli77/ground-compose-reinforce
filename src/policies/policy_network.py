import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from gymnasium.spaces import Box, Discrete


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, action_space, hiddens=[], layer_norm=False, activation=nn.Tanh()):
        super().__init__()

        layer_dims = [in_dim] + hiddens
        self.action_space = action_space
        self.num_layers = len(layer_dims)
        self.enc_ = nn.Sequential(*[fc(in_dim, out_dim, layer_norm, activation=activation)
            for (in_dim, out_dim) in zip(layer_dims, layer_dims[1:])])

        if (isinstance(self.action_space, Discrete)):
            action_dim = self.action_space.n
            self.out_layer = nn.Linear(layer_dims[-1], action_dim)
        elif (isinstance(self.action_space, Box)):
            assert((self.action_space.low == -1).all())
            assert((self.action_space.high == 1).all())
            action_dim = torch.prod(torch.tensor(self.action_space.shape))

            self.out_layer = nn.Linear(layer_dims[-1], action_dim*2)
            self.std_scale = nn.Softplus()
        else:
            print("Unsupported action_space type: ", self.action_space)
            exit(1)

    def forward(self, obs):
        if (isinstance(self.action_space, Discrete)):
            x = self.enc_(obs)
            x = self.out_layer(x)
            return Categorical(logits=F.log_softmax(x, dim=1))
        elif (isinstance(self.action_space, Box)):
            x = self.enc_(obs)
            out = self.out_layer(x)
            mu, logstd = torch.split(out, out.shape[-1]//2, dim=-1)
            # new_shape = (-1,) + self.action_space.shape
            return Normal(mu, self.std_scale(logstd))
        else:
            print("Unsupported action_space type: ", self.action_space)
            exit(1)


def fc(in_dim, out_dim, layer_norm=False, activation=nn.Tanh()):
    if layer_norm:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            activation
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation
        )