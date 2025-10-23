"""
Given a dataset of trajectories labelled with propositions, we fit the following foundation models:

1) A labelling function $L(s) -> 2^{AP}$. Maps the current state to the atomic propositions that are true in that state.
2) A value function, which is either
	- atomic: $V_p(s)$, which predicts the current state value for the task "eventually p", where p is a literal.
	- conjunctive: $V_phi(s)$, which predicts the current state value for the task "eventually phi", where phi is a conjunctive of literals.
"""
import os

import numpy as np, torch
import src.envs as envs
from src.vlm_models import *
"""
================== LOAD CONFIG ==================
"""
from config import CONFIG, SETTINGS
env_name = SETTINGS['env']
assert env_name == "grid-env"
config = CONFIG[env_name]

batch_size = config['offline_rl']['batch_size']
gamma = config['rl_train']['rm_discount']
lr = config['offline_rl']['lr']
max_epochs = config['offline_rl']['max_epochs']
n_props = 5
n_values = 19
validation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
================== LOAD DATA ==================
""" 

print("================Loading data================")
data_dir = CONFIG['grid-env']['offline_rl']['data_dir']
_obss = np.float32(np.load(os.path.join(data_dir, 'obss.npz'))['arr_0'])
_actions = np.int32(np.load(os.path.join(data_dir, 'actions.npz'))['arr_0'])
_labels = np.load(os.path.join(data_dir, 'labels.npz'))['arr_0']
_labels = np.concatenate((
	_labels,
	1-_labels,
	np.expand_dims(_labels[:,:,0] & _labels[:,:,3], -1),
	np.expand_dims(_labels[:,:,0] & _labels[:,:,4], -1),
	np.expand_dims(_labels[:,:,1] & _labels[:,:,3], -1),
	np.expand_dims(_labels[:,:,1] & _labels[:,:,4], -1),
	np.expand_dims(_labels[:,:,2] & _labels[:,:,3], -1),
	np.expand_dims(_labels[:,:,2] & _labels[:,:,4], -1),
	np.expand_dims(_labels[:,:,0] & 1 - _labels[:,:,4], -1),
	np.expand_dims(_labels[:,:,1] & 1 - _labels[:,:,4], -1),
	np.expand_dims(_labels[:,:,2] & 1 - _labels[:,:,4], -1),
	), axis=-1
)

# Processed data
obs = _obss[:, :-1]
next_obs = _obss[:, 1:]
actions = _actions
labels = _labels[:, :-1]
next_labels = _labels[:,1:]

if validation:
	n_total = obs.shape[0]
	n_validation = round(obs.shape[0] * 0.1)

	train_dataset = torch.utils.data.TensorDataset(*(torch.tensor(tnsor)[:n_total - n_validation].flatten(end_dim=1).to(device) for tnsor in [obs, next_obs, actions, labels, next_labels]))
	train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	validation_dataset = torch.utils.data.TensorDataset(*(torch.tensor(tnsor)[-n_validation:].flatten(end_dim=1).to(device) for tnsor in [obs, next_obs, actions, labels, next_labels]))

else:
	train_dataset = torch.utils.data.TensorDataset(*(torch.tensor(tnsor).flatten(end_dim=1).to(device) for tnsor in [obs, next_obs, actions, labels, next_labels]))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""
================== LOAD MODELS ==================
"""

print("================Loading models================")
foundation_model = get_offline_foundation_model(env_name, device)
progress_optimizer = torch.optim.Adam(foundation_model.progress.parameters(), lr=3e-4) # , weight_decay=1e-6)
q_optimizer = torch.optim.Adam(foundation_model.q.parameters(), lr=3e-4) # , weight_decay=1e-6)

vf_savefile = os.path.join(data_dir, "progress.pt")

"""
================== TRAINING ==================
"""

@torch.no_grad()
def evaluate_progress():

	obs = validation_dataset.tensors[0]
	next_obs = validation_dataset.tensors[1]
	action = validation_dataset.tensors[2]
	labels = validation_dataset.tensors[3]
	next_labels = validation_dataset.tensors[4]

	mask = 1 - labels
	nnz = mask.sum() 

	mask_next = 1 - next_labels
	nnz_next = mask_next.sum()

	# Update progress model
	q_targets, v_targets = get_qv_targets(obs, next_obs, action, labels, next_labels)
	qs = foundation_model.get_qs(obs).reshape(obs.shape[0], 4, -1)
	q_pred = torch.gather(qs, dim=1, index=action.long().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 19)).squeeze(1)
	v_pred = foundation_model.get_progress(next_obs)

	loss_q = ((q_pred - q_targets)**2 * mask).sum() / nnz
	loss_v = ((v_pred - v_targets)**2 * mask_next).sum() / nnz_next

	return loss_q.item(), loss_v.item()

@torch.no_grad()
def get_qv_targets(obs, next_obs, action, labels, next_labels):
	q_targets = []
	v_targets = []

	next_vs = foundation_model.get_qs(next_obs).reshape(obs.shape[0], 4,-1).max(dim=1).values
	r = next_labels.float()
	done = (r > 0).float()

	return r + gamma * (1-done) * next_vs, next_vs

# Trains both the classifier and progress models.
def train(evaluation_frequency=10):
	for epoch in range(max_epochs):
		train_loss_q = []
		train_loss_v = []

		for obs, next_obs, action, labels, next_labels in train_loader:
			mask = 1 - labels
			nnz = mask.sum() 

			mask_next = 1 - next_labels
			nnz_next = mask_next.sum()

			# Update progress model
			with torch.no_grad():
				q_targets, v_targets = get_qv_targets(obs, next_obs, action, labels, next_labels)
			
			qs = foundation_model.get_qs(obs).reshape(obs.shape[0], 4, -1)
			q_pred = torch.gather(qs, dim=1, index=action.long().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 19)).squeeze(1)
			v_pred = foundation_model.get_progress(next_obs)

			loss_q = ((q_pred - q_targets)**2 * mask).sum() / nnz
			loss_v = ((v_pred - v_targets)**2 * mask_next).sum() / nnz_next

			progress_optimizer.zero_grad()
			loss_v.backward()
			progress_optimizer.step()

			q_optimizer.zero_grad()
			loss_q.backward()
			q_optimizer.step()

			train_loss_q.append(loss_q.item())
			train_loss_v.append(loss_v.item())

		q_loss_val, v_loss_val = evaluate_progress() if validation else 0,0
		print("TRAIN ---- Epoch %d -- Q loss: %.5f -- V loss: %.5f -- Val Q loss: %.5f -- Val V loss: %.5f" % (epoch, torch.tensor(train_loss_q).mean(), torch.tensor(train_loss_v).mean(), q_loss_val, v_loss_val))

	status = {
		"progress_model": foundation_model.progress.state_dict(),
	}
	torch.save(status, vf_savefile)

train()
