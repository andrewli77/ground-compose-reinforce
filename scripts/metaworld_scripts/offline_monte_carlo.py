"""
Given a dataset of trajectories labelled with propositions, we fit one of the following value functions:
    - atomic: $V_p(s)$, which predicts the current state value for the task "eventually p", where p is a literal.
    - conjunctive: $V_phi(s)$, which predicts the current state value for the task "eventually phi", where phi is a conjunctive of literals.
"""
import os, time

import numpy as np, torch
import src.envs as envs
from src.vlm_models import *
from torch.distributions import Normal, kl_divergence

def load_data(data_dir, gamma, validation=False):
    obss_raw = np.load(os.path.join(data_dir, 'obss.npz'))
    labels_raw = np.load(os.path.join(data_dir, 'labels.npz'))

    all_distances = []

    for i in range(len(labels_raw)):
        _labels = labels_raw[f'arr_{i}']

        labels_i = np.concatenate([
            _labels,
            1 - _labels,
            np.expand_dims(_labels[:,6] & _labels[:,0], axis=-1), # gbox_in_drawer1&drawer1_open
            np.expand_dims(_labels[:,9] & _labels[:,1], axis=-1), # gbox_in_drawer2&drawer2_open
            np.expand_dims((1 - _labels[:,6]) & (1 - _labels[:,9]), axis=-1), # !gbox_in_drawer1&!gbox_in_drawer2
            np.expand_dims(_labels[:,6] & (1 - _labels[:,0]), axis=-1), # gbox_in_drawer1&!drawer1_open
            np.expand_dims(_labels[:,9] & (1 - _labels[:,1]), axis=-1), # gbox_in_drawer2&!drawer2_open
        ], axis=1)

        distances = np.full(labels_i.shape, np.inf)

        for j in reversed(range(len(labels_i) - 1)):
            distances[j] = distances[j+1] + 1
            distances[j][labels_i[j+1] == 1] = 0.

        all_distances.append(distances)

    n_total = len(obss_raw)

    if validation:
        import random
        random.seed(1)
        n_validation = round(len(obss_raw)*0.1)
        validation_idxs = set(random.sample(range(n_total), n_validation))

        obss = np.concatenate([obss_raw[f'arr_{k}'][:-2] for k in range(n_total) if k not in validation_idxs], axis=0, dtype=np.float32)
        dists = np.concatenate([all_distances[k][:-1] for k in range(n_total) if k not in validation_idxs], axis=0, dtype=np.float32)
        values = gamma ** dists

        val_obss = np.concatenate([obss_raw[f'arr_{k}'][:-2] for k in range(n_total) if k in validation_idxs], axis=0, dtype=np.float32)
        val_dists = np.concatenate([all_distances[k][:-1] for k in range(n_total) if k in validation_idxs], axis=0, dtype=np.float32)
        val_values = gamma ** val_dists

        return (obss[:, :39], values), (val_obss[:, :39], val_values), 27
    else:
        obss = np.concatenate([obss_raw[f'arr_{k}'][:-2] for k in range(n_total)], axis=0, dtype=np.float32)
        dists = np.concatenate([all_distances[k][:-1] for k in range(n_total)], axis=0, dtype=np.float32)
        values = gamma ** dists
        return (obss[:, :39], values), 27


if __name__ == '__main__':
    """
    ================== LOAD CONFIG ==================
    """
    from config import CONFIG, SETTINGS
    env_name = SETTINGS['env']
    assert env_name == 'drawer-env'
    config = CONFIG[env_name]
    data_dir = config['offline_rl']['data_dir']
    batch_size = config['offline_rl']['batch_size']
    gamma = config['offline_rl']['rm_discount']
    max_epochs = config['offline_rl']['max_epochs']
    pseudo_obs_lambda = config['offline_rl']['pseudo_obs_lambda']
    validation = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # """
    # ================== LOAD DATA ==================
    # """
    # Transform the data into samples of the form (s, a, s', L(s))
    
    print("================Loading data================")
    if validation:
        data, val_data, n_dists = load_data(data_dir, gamma, validation=True)
        validation_dataset = torch.utils.data.TensorDataset(*(torch.tensor(tnsor).to(device) for tnsor in val_data))
    else:
        data, n_dists = load_data(data_dir, gamma, validation=False)
    
    dataset = torch.utils.data.TensorDataset(*(torch.tensor(tnsor).to(device) for tnsor in data))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Add obs statistics so we can generate random pseudo-observations
    mean_obs = torch.tensor(data[0].mean(axis=0), device=device)
    std_obs = torch.tensor(data[0].std(axis=0), device=device)

    """
    ================== LOAD MODELS ==================
    """

    print("================Loading models================")
    foundation_model = get_offline_foundation_model(env_name, device)
    progress_optimizer = torch.optim.Adam(foundation_model.progress.parameters(), lr=3e-4) # , weight_decay=1e-6)

    """
    ================== TRAINING ==================
    """

    def evaluate(validation_dataset):
        obss, values = validation_dataset.tensors

        with torch.no_grad():
            foundation_model.eval()
            preds = foundation_model.get_progress(obss) 
            loss = (preds - values).pow(2).mean().item()

        return loss

    def train(data_loader, log_frequency=1):
        for epoch in range(1, max_epochs + 1):
            progress_losses = []
            time_epoch_start = time.time()

            for obs, values in data_loader:
                # Train progress model
                progress_preds = foundation_model.get_progress(obs)

                # Generate pseudo-observations for conservatism.
                # The target value for these is 0.
                ood_obs = mean_obs + std_obs * torch.randn_like(obs)
                ood_preds = foundation_model.get_progress(ood_obs)

                progress_loss = (progress_preds - values).pow(2).mean() + pseudo_obs_lambda * ood_preds.pow(2).mean()
                l1_loss = sum(torch.sum(torch.abs(param)) for param in foundation_model.progress.parameters())

                progress_losses.append(progress_loss.item())
                progress_optimizer.zero_grad()
                (progress_loss + 1e-5 * l1_loss).backward()
                progress_optimizer.step()

            time_epoch_end = time.time()

            if validation:
                val_loss = evaluate(validation_dataset)
            else:
                val_loss = 0.

            if epoch % log_frequency == 0:
                print("Epoch: %d -- Duration: %d -- Loss: %.5f -- Val Loss: %.5f" %(epoch, int(time_epoch_end - time_epoch_start), torch.tensor(progress_losses).mean().item(), val_loss))

        status = {
            "progress_model": foundation_model.progress.state_dict()
        }

        torch.save(status, os.path.join(data_dir, "progress.pt"))

    train(data_loader)
