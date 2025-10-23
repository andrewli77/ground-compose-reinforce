import torch
import torch.nn as nn
import numpy as np
from config import CONFIG, SETTINGS, device

# # ---------------- Ground truth models ----------------


class GridEnvGTFoundationModel():
    def query(self, obs, info=None):
        if type(info)==tuple:
            labels = torch.tensor(np.array([i['labels'] for i in info]), device=device)
            dists = torch.tensor(np.array([i['dists'] for i in info]), device=device)
            return labels, torch.pow(CONFIG['grid-env']['rl_train']['rm_discount'], dists)
        else:
          return torch.tensor(info['labels'], device=device).int(), torch.tensor(np.power(CONFIG['grid-env']['rl_train']['rm_discount'], info['dists']), device=device)


# ---------------- OfflineRL models ----------------

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class GridEnvFoundationModel(nn.Module):
    def __init__(self, num_props, num_values):
        super().__init__()
        # [r,g,b,c,t]
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, num_props)
        )

        # Progresses: [r,g,b,c,t,!r,!g,!b,!c,!t] + [rc, rt, gc, gt, bc, bt, r&!t, g&!t, b&!t]

        self.progress = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, num_values)
        )

        self.q = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 4*num_values)
        )

        self.apply(init_params)

    def forward(self, obs):
        return self.classifier(obs.permute(0,3,1,2)), self.progress(obs.permute(0,3,1,2))

    @torch.no_grad()
    def query(self, obs, info=None):
        clss = self.classifier(obs.permute(0,3,1,2)) > 0
        progress = self.progress(obs.permute(0,3,1,2))
        return clss, progress

    def get_classification(self, obs):
        return self.classifier(obs.permute(0,3,1,2))

    def get_progress(self, obs):
        return self.progress(obs.permute(0,3,1,2))

    def get_qs(self, obs):
        return self.q(obs.permute(0,3,1,2))

# class GridCNNClassifierModel(ClassifierModel, nn.Module):
#     def __init__(self, obs_space, num_outputs):
#         super().__init__()

#         self.conv_net = nn.Sequential(
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(32*8*8, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_outputs)
#         )

#         self.apply(init_params)

#     def forward(self, obs):
#         image_encoding = self.conv_net(obs.permute(0,3,1,2))
#         return self.classifier(image_encoding)

#     @torch.no_grad()
#     def __call__(self, obs, info=None):
#         image_encoding = self.conv_net(obs.permute(0,3,1,2))
#         preds = self.classifier(image_encoding)
#         return tuple([self.pred_to_true_props(preds[i]) for i in range(preds.shape[0])])

#     def pred_to_true_props(self, pred):
#         assert len(pred.shape)==1 
#         true_props = ""
#         if pred[0] > 0:
#             true_props += 'r'
#         if pred[1] > 0:
#             true_props += 'g'
#         if pred[2] > 0:
#             true_props += 'b'
#         if pred[3] > 0:
#             true_props += 'c'
#         if pred[4] > 0:
#             true_props += 't'
#         return true_props

# class GridCNNProgressModel(ProgressModel, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs):
#         super().__init__()

#         self.n_actions = action_space.n
#         self.num_outputs = num_outputs

#         self.conv_net = nn.Sequential(
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )

#         self.q = nn.Sequential(
#             nn.Linear(32*8*8 + action_space.n, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_outputs)
#         )


#         self.v = nn.Sequential(
#             nn.Linear(32*8*8, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_outputs)
#         )

#         self.apply(init_params)

#     def forward(self, obs, action):
#         image_encoding = self.conv_net(obs.permute(0,3,1,2))
#         q = self.q(torch.cat([image_encoding, F.one_hot(action, num_classes=self.n_actions)], dim=1))
#         v = self.v(image_encoding)
#         return q,v

#     def forward_v(self, obs):
#         image_encoding = self.conv_net(obs.permute(0,3,1,2))
#         v = self.v(image_encoding)
#         return v

#     @torch.no_grad()
#     def __call__(self, obs, info=None):
#         dists = self.forward_v(obs)
#         return tuple([self.get_dists(dists[i]) for i in range(dists.shape[0])])

#     def get_dists(self, array):
#         if self.num_outputs == 5:
#             return {'r': array[0], 'g': array[1], 'b': array[2], 'c': array[3], 't': array[4]}
#         else:
#             return {'r': array[0], 'g': array[1], 'b': array[2], 'c': array[3], 't': array[4], 'r&c': array[5], 'r&t': array[6], 'g&c': array[7], 'g&t': array[8], 'b&c': array[9], 'b&t': array[10]}