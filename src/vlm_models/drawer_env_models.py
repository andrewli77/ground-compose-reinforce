import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from config import CONFIG, SETTINGS, device

# ---------------- OfflineRL models ----------------

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class DrawerEnvFoundationModel(nn.Module):
    def __init__(self, num_props, num_values):
        super().__init__()

        gamma = CONFIG['drawer-env']['rl_train']['rm_discount']
        average_transition_length = CONFIG['drawer-env']['rl_train']['average_transition_length']
        self.min_value = torch.tensor(gamma, device=device) ** average_transition_length  

        # 'drawer1_open', 'drawer2_open', 'rbox_lifted', 'gbox_lifted', 'bbox_lifted', 'rbox_in_drawer1', 'gbox_in_drawer1', 'bbox_in_drawer1', 'rbox_in_drawer2', 'gbox_in_drawer2', 'bbox_in_drawer2'
        self.classifier = nn.Sequential(
            nn.Linear(39, 1600),
            nn.ReLU(),
            nn.Linear(1600, num_props)
        )
        # [all_propositions], [all negated propositions], gbox_in_drawer1&drawer1_open, gbox_in_drawer2&drawer2_open, !gbox_in_drawer1&!gbox_in_drawer2, gbox_in_drawer1&!drawer1_open, gbox_in_drawer2&!drawer2_open
        self.progress = nn.Sequential(
            nn.Linear(39, 1600),
            nn.ReLU(),
            nn.Linear(1600, 400),
            nn.ReLU(),
            nn.Linear(400, num_values)
        )

        self.apply(init_params)

    def forward(self, obs):
        return self.classifier(obs), self.progress(obs)

    # Smoothly clamp values into the range [min_value, 1]
    def soft_clamp(self, v, alpha=3.):
        return self.min_value + (1 - self.min_value) * (1 - torch.exp(-alpha * v)) / (1 - torch.exp(torch.tensor(-alpha, device=device)))

    @torch.no_grad()
    def query(self, obs, info=None):
        clss = self.classifier(obs[:, :39]) > 0
        progress = self.soft_clamp(self.progress(obs[:,:39]))
        return clss, progress

    def get_classification(self, obs):
        return self.classifier(obs)

    def get_progress(self, obs):
        return self.progress(obs)
