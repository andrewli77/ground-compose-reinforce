# Visualize the policy learned by RL or from offline RL

import torch 
from src.envs import rm_env_constructor
from scripts.metaworld_scripts.manual_control import *
import src.utils as utils
from src.vlm_models import get_offline_foundation_model
from config import CONFIG, SETTINGS

from src.envs.gridworld import ActionGUI

# Display options
PRINT_GROUND_TRUTH_LABELS=0
PRINT_PREDICTED_LABELS=0
PRINT_PREDICTED_PROGRESS=0
PRINT_PREDICTED_RM_INFO=1

# Load config
assert SETTINGS['env'] == 'grid-env'
TASK = SETTINGS['rm_file']
rm_task_name = TASK.split('/')[-1].removesuffix(".txt")
gamma = CONFIG['grid-env']['rl_train']['rm_discount']
average_transition_length = CONFIG['grid-env']['rl_train']['average_transition_length']
rm_algo = CONFIG['grid-env']['rl_train']['rm_algo']

from src.policies import get_acmodel
ac_model = get_acmodel("grid-env", TASK) 
policy_weights_dir = f"storage/grid-env/offline_rl/rl_train/{rm_task_name}/{rm_algo}/1/"
status = utils.get_status(policy_weights_dir)
ac_model.load_state_dict(status["model_state"])

env = rm_env_constructor('grid-env', TASK, gamma, average_transition_length, rm_algo)()
preprocess_obss = utils.get_obss_preprocessor("grid-env")
foundation_model = get_offline_foundation_model("grid-env", load_weights=True)

gui = ActionGUI(env, foundation_model, PRINT_GROUND_TRUTH_LABELS, PRINT_PREDICTED_LABELS, PRINT_PREDICTED_PROGRESS, PRINT_PREDICTED_RM_INFO)

o, info = gui.reset()

while True:
    with torch.no_grad():
        preprocessed_obs = preprocess_obss([o])
        dist, _ = ac_model(preprocessed_obs)
        action = dist.sample()

    o, done = gui.step(action.numpy())   

    if done:
        break
