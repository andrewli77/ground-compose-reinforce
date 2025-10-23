"""
Given a dataset of trajectories labelled with propositions, we fit the following foundation models:

1) A labelling function $L(s) -> 2^{AP}$. Maps the current state to the atomic propositions that are true in that state.
2) A value function, which is either
    - atomic: $V_p(s)$, which predicts the current state value for the task "eventually p", where p is a literal.
    - conjunctive: $V_phi(s)$, which predicts the current state value for the task "eventually phi", where phi is a conjunctive of literals.
"""
import os, time

import numpy as np, torch
import src.envs as envs
import torch.nn.functional as F
from src.vlm_models import *

def load_data_drawer(data_dir, validation=False):
    obss_raw = np.load(os.path.join(data_dir, 'obss.npz'))
    actions_raw = np.load(os.path.join(data_dir, 'actions.npz'))
    labels_raw = np.load(os.path.join(data_dir, 'labels.npz'))

    n_total = len(obss_raw)
    if validation:
        import random
        random.seed(1)

        n_validation = round(len(obss_raw)*0.1)
        validation_idxs = set(random.sample(range(n_total), n_validation))

        obss = np.concatenate([obss_raw[f'arr_{k}'][:-2] for k in range(n_total) if k not in validation_idxs], axis=0, dtype=np.float32)
        labels = np.concatenate([labels_raw[f'arr_{k}'][:-1] for k in range(n_total) if k not in validation_idxs], axis=0, dtype=np.uint8)

        val_obss = np.concatenate([obss_raw[f'arr_{k}'][:-2] for k in range(n_total) if k in validation_idxs], axis=0, dtype=np.float32)
        val_labels = np.concatenate([labels_raw[f'arr_{k}'][:-1] for k in range(n_total) if k in validation_idxs], axis=0, dtype=np.uint8)

        return (obss[:, :39], labels), (val_obss[:, :39], val_labels), 11, 22 # Include only first 39 dimensions since the last 39 dimensions are for frame stacking

    else:
        obss = np.concatenate([obss_raw[f'arr_{k}'][:-2] for k in range(n_total)], axis=0, dtype=np.float32)
        labels = np.concatenate([labels_raw[f'arr_{k}'][:-1] for k in range(n_total)], axis=0, dtype=np.uint8)
        return (obss[:, :39], labels), 11, 22

def load_data_grid(data_dir, validation=False):
    obss_raw = np.load(os.path.join(data_dir, 'obss.npz'))
    actions_raw = np.load(os.path.join(data_dir, 'actions.npz'))
    labels_raw = np.load(os.path.join(data_dir, 'labels.npz'))

    n_total = obss_raw['arr_0'].shape[0]
    if validation:
        import random
        random.seed(1)

        n_validation = round(n_total*0.1)

        obss = obss_raw['arr_0'][:n_total - n_validation]
        obss = obss.reshape(obss.shape[0]*obss.shape[1],*obss.shape[2:])

        labels = labels_raw['arr_0'][:n_total - n_validation]
        labels = labels.reshape(labels.shape[0]*labels.shape[1],*labels.shape[2:])

        val_obss = obss_raw['arr_0'][-n_validation:]
        val_obss = val_obss.reshape(val_obss.shape[0]*val_obss.shape[1],*val_obss.shape[2:])

        val_labels = labels_raw['arr_0'][-n_validation:]
        val_labels = val_labels.reshape(val_labels.shape[0]*val_labels.shape[1],*val_labels.shape[2:])

        return (obss, labels), (val_obss, val_labels), 5, 19

    else:
        obss = obss_raw['arr_0']
        obss = obss.reshape(obss.shape[0]*obss.shape[1],*obss.shape[2:])
        labels = labels_raw['arr_0']
        labels = labels.reshape(labels.shape[0]*labels.shape[1],*labels.shape[2:])
        
        return (obss, labels), 5, 19


"""
================== LOAD CONFIG ==================
"""
from config import CONFIG, SETTINGS
env_name = SETTINGS['env']
config = CONFIG[env_name]
data_dir = config['offline_rl']['data_dir']
batch_size = config['classifier_train']['batch_size']
max_epochs = config['classifier_train']['max_epochs']
validation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# """
# ================== LOAD DATA ==================
# """
# Transform the data into samples of the form (s, a, s', L(s))

print("================Loading data================")
if validation:
    if env_name == 'grid-env':
        data, validation_data, n_props, n_dists = load_data_grid(data_dir, validation=True)
    elif env_name == 'drawer-env':
        data, validation_data, n_props, n_dists = load_data_drawer(data_dir, validation=True)
    validation_dataset = torch.utils.data.TensorDataset(*(torch.tensor(tnsor).to(device) for tnsor in validation_data))
else:
    if env_name == 'grid-env':
        data, n_props, n_dists = load_data_grid(data_dir, validation=False)
    elif env_name == 'drawer-env':
        data, n_props, n_dists = load_data_drawer(data_dir, validation=False)

dataset = torch.utils.data.TensorDataset(*(torch.tensor(tnsor).to(device) for tnsor in data))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

"""
================== LOAD MODELS ==================
"""

print("================Loading models================")
foundation_model = get_offline_foundation_model(env_name, device=device)
classifier_optimizer = torch.optim.Adam(foundation_model.classifier.parameters(), lr=3e-4)

"""
================== TRAINING ==================
"""

def evaluate(validation_dataset):
    obss, labels = validation_dataset.tensors

    with torch.no_grad():
        foundation_model.eval()
        clss = foundation_model.get_classification(obss) 
        loss = F.binary_cross_entropy_with_logits(clss, labels[:, :n_props].float())
        accuracy = ((clss > 0) == labels[:, :n_props]).float().mean()

    return loss, accuracy

def train(data_loader, log_frequency=1):
    for epoch in range(1, max_epochs + 1):
        classifier_losses = []

        time_epoch_start = time.time()

        for obs, labels in data_loader:

            # Train clasifier model
            foundation_model.train()
            classifier_preds = foundation_model.get_classification(obs)            
            classifier_loss = F.binary_cross_entropy_with_logits(classifier_preds, labels[:, :n_props].float())

            l1_loss = sum(torch.sum(torch.abs(param)) for param in foundation_model.classifier.parameters())

            classifier_losses.append(classifier_loss.item())
            classifier_optimizer.zero_grad()
            (classifier_loss + 1e-5 * l1_loss).backward()
            classifier_optimizer.step()

        time_epoch_end = time.time()

        if validation:
            val_loss, val_acc = evaluate(validation_dataset)
        else:
            val_loss, val_acc = torch.tensor(0.0), torch.tensor(0.0)

        if epoch % log_frequency == 0:
            print("Epoch: %d -- Duration: %d -- Train loss: %.5f -- Val loss: %.5f -- Val accuracy: %.5f" %((epoch, int(time_epoch_end - time_epoch_start), torch.tensor(classifier_losses).mean().item(), val_loss.item(), val_acc.item())))

    status = {
        "classifier_model": foundation_model.classifier.state_dict()
    }
    torch.save(status, os.path.join(data_dir, f"classifier.pt"))

train(data_loader)
