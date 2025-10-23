# Visualize the policy learned by RL or from offline RL

import torch 
from src.envs import rm_env_constructor
from scripts.metaworld_scripts.manual_control import *
import src.utils as utils
from src.vlm_models import get_offline_foundation_model
from config import CONFIG, SETTINGS



# Display options
PRINT_GROUND_TRUTH_LABELS=0
PRINT_PREDICTED_LABELS=0
PRINT_PREDICTED_PROGRESS=0
PRINT_PREDICTED_RM_INFO=1

# Load config
assert SETTINGS['env'] == 'drawer-env'
TASK = SETTINGS['rm_file']
rm_task_name = TASK.split('/')[-1].removesuffix(".txt")
gamma = CONFIG['drawer-env']['rl_train']['rm_discount']
average_transition_length = CONFIG['drawer-env']['rl_train']['average_transition_length']
rm_algo = CONFIG['drawer-env']['rl_train']['rm_algo']

from src.policies import get_acmodel
ac_model = get_acmodel("drawer-env", TASK) 
policy_weights_dir = f"storage/drawer-env/offline_rl/rl_train/{rm_task_name}/{rm_algo}/1/"
status = utils.get_status(policy_weights_dir)
ac_model.load_state_dict(status["model_state"])

env = PlayWrapper(rm_env_constructor('drawer-env', TASK, gamma, average_transition_length, rm_algo)())
preprocess_obss = utils.get_obss_preprocessor("drawer-env")
foundation_model = get_offline_foundation_model("drawer-env", load_weights=True)

o, info = env.reset()
classifications, progresses = foundation_model.query(torch.tensor(o['observation'], dtype=torch.float32).unsqueeze(0))
env.env.initialize_potential(classifications[0], progresses[0])

print("Initial potential:", torch.tensor(env.env.current_potential).item())

step = 0 

while True:
    env.render()
    
    with torch.no_grad():
        preprocessed_obs = preprocess_obss([o])
        dist, _ = ac_model(preprocessed_obs)
        action = dist.sample().tanh()[0]

    classifications, progresses = foundation_model.query(torch.tensor(o['observation'], dtype=torch.float32).unsqueeze(0))
    rm_reward, potential_reward, rm_done = env.env.sync(classifications[0], progresses[0])

    print(env.env.current_true_u_id, env.env.current_agent_u_id)

    o, r, term, trunc, new_info = env.step(action.numpy())   

    if PRINT_GROUND_TRUTH_LABELS:
        print("Ground truth labels:", new_info['labels'])
    if PRINT_PREDICTED_LABELS:
        print("Predicted labels:", pp(classifications[0]))
    if PRINT_PREDICTED_PROGRESS:
        print("Predicted progress:", pp(progresses[0]))
    if PRINT_PREDICTED_RM_INFO:
        print("Step:", step, "Reward machine state:", o['rm_state'], "RM potential:", torch.tensor(env.env.current_potential).item())

    if (term or trunc):
        o, info = env.reset()
        print(info)

    step += 1