from config import CONFIG, SETTINGS
from src.vlm_models import get_foundation_model
from src.envs import rm_env_constructor
from src.envs.gridworld import InteractiveGUI
import torch, numpy as np

# Set display options here
PRINT_GROUND_TRUTH_LABELS=0
PRINT_PREDICTED_LABELS=0
PRINT_PREDICTED_PROGRESS=0
PRINT_PREDICTED_RM_INFO=1

# Load config
ENV = SETTINGS['env']
TASK = SETTINGS['rm_file']
MODEL_TYPE = SETTINGS['vlm_model_type']
rm_discount = CONFIG[ENV]['rl_train']['rm_discount']
average_transition_length = CONFIG[ENV]['rl_train']['average_transition_length']
rm_algo = CONFIG[ENV]['rl_train']['rm_algo']

assert ENV == "grid-env"

# Load foundation models
foundation_model = get_foundation_model(ENV, MODEL_TYPE)

# Load env
env = rm_env_constructor(ENV, TASK, rm_discount, average_transition_length, rm_algo)()


if SETTINGS['env'] == 'grid-env':
    gui = InteractiveGUI(env, foundation_model, PRINT_GROUND_TRUTH_LABELS, PRINT_PREDICTED_LABELS, PRINT_PREDICTED_PROGRESS, PRINT_PREDICTED_RM_INFO)
    gui.run()