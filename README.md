# Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data

This is the official repository for our **NeurIPS 2025** paper [Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data](https://arxiv.org/pdf/2507.10741). 

Our approach enables end-to-end RL without oracles like reward functions or feature extractors. Instead, it starts with a small number of trajectories labelled with propositional symbols, and recomposes these symbols into arbitrary Reward Machine tasks. 

<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/b9aac3bd-c3ee-48a8-a5b5-ba8117f1f7c4" width="200"><br><b>Hold red block as long as possible</b></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/f760cb84-61b2-4e76-9535-df374d651e44" width="200"><br><b>Pick up each block</b></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/3b32bc52-3f05-4321-8f75-12923bac52af" width="200"><br><b>Reveal and pick up the green block</b></td>
  </tr>
</table>


## Setup

All code was tested with the following settings: Python 3.9, PyTorch 2.0.1, NumPy 1.26.4, gymnasium 1.0.0a2, with and without GPU acceleration (CUDA 11.7). We use modified versions of the following libraries (directly included inside our code), with thanks to their respective developers: [torch-ac](https://github.com/lcswillems/torch-ac), [Reward Machines](https://github.com/RodrigoToroIcarte/reward_machines), [Meta-World](https://github.com/Farama-Foundation/Metaworld). 

- We include our compressed pretraining datasets (with labels) and trained models in the `storage/` directory for future research efforts. 
- Our code can be run with minimal changes (e.g. running `python scripts/metaworld_scripts/visualize_policy.py` should execute a trained policy).
- Most scripts are easily configurable just by changing the `env` and `rm_task` options in `config.py` (hyperparameter settings are also found in this file). 

**IMPORTANT**: Remember to correctly set `env` and `rm_task` in `config.py` before running any scripts. Run code from this directory (set `export PYTHONPATH=.` to make sure all modules load properly).  

## Manually play with the environments.
- Meta-World: `python scripts/metaworld_scripts/manual_control.py` (see header comments for keyboard controls)
- GridWorld: `python scripts/gridworld_scripts/manual_control.py` (WASD controls)

## Record trajectory data.
- Meta-World: `python scripts/metaworld_scripts/record_trajectories.py` (manually control the robot to record data; see header comments for keyboard controls)
- GridWorld: `python3 scripts/gridworld_scripts/record_trajectories.py` (random-action policy collects the data)

## Pretrain propositional labelling functions from data.
- `python scripts/train_classifier.py`

## Pretrain value function estimators from data.
- Meta-World: `python scripts/metaworld_scripts/offline_monte_carlo.py`
- GridWorld: `python scripts/gridworld_scripts/offline_ql.py`

## Train agent with RL directly from an RM specification.
- `python scripts/rl_train.py`

