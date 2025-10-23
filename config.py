"""
SETTINGS provides an easy way to set the variables of CONFIG.
CONFIG maps each environment name to the hyperparameter settings 
"""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SETTINGS = {
	'env': 'grid-env', # ['grid-env', 'drawer-env']
	'rm_task': 'loop', # see ALL_TASKS for options
}

CONFIG = {}

"""
================== Automated checks ==================
""" 
ALL_TASKS = {
	'drawer-env': ['lift_red_box', 'pickup_each_box', 'show_green'],
	'grid-env': ['logic', 'loop', 'sequence', 'sequence_safety']	
}
assert SETTINGS['rm_task'] in ALL_TASKS[SETTINGS['env']]
SETTINGS['rm_file'] = f"src/reward_machines/rm_files/{SETTINGS['env']}/{SETTINGS['rm_task']}.txt"
SETTINGS['vlm_model_type'] = 'offline_rl'

"""
================== GRID ENVIRONMENT ==================
""" 
CONFIG['grid-env'] = {}

# classifier learning parameters
CONFIG['grid-env']['classifier_train'] = {
	'data_dir': 'storage/grid-env/offline_rl',
	'batch_size': 256,
	'max_epochs': 10,
}

# offline RL value function learning parameters
CONFIG['grid-env']['offline_rl'] = {
	'data_dir': 'storage/grid-env/offline_rl',
	'batch_size': 1024,
	'lr': 0.0003,
	'max_epochs': 100,
	'rm_discount': 0.97
}

# online RL training parameters
CONFIG['grid-env']['rl_train'] = {
	'rm_discount': 0.97,
	'average_transition_length': 10,
	'rm_sync_freq': 1,
	'rm_algo': 'atomic', # none, u, atomic, conjunctive
}

for task in ['sequence', 'loop', 'logic', 'sequence_safety']:
	CONFIG['grid-env']['rl_train'][task] = {
		'epochs': 4,
		'batch_size': 4000,
		'frames_per_proc': 1000,
		'discount': 0.97,
		'gae_lambda': 0.95,
		'potential_scale': 1.,
		'lr': 0.0003,
		'entropy_coef': 0.0001,
	}


"""
================== DRAWER ENVIRONMENT ==================
""" 
CONFIG['drawer-env'] = {}

# classifier learning parameters
CONFIG['drawer-env']['classifier_train'] = {
	'data_dir': 'storage/drawer-env/offline_rl',
	'batch_size': 256,
	'max_epochs': 100,
}

# offline RL value function learning parameters
CONFIG['drawer-env']['offline_rl'] = {
	'data_dir': 'storage/drawer-env/offline_rl',
	'batch_size': 256,
	'max_epochs': 100,
	'rm_discount': 0.9975,
	'pseudo_obs_lambda': 0.005
}

# online RL training parameters
CONFIG['drawer-env']['rl_train'] = {
	'rm_discount': 0.9975,
	'average_transition_length': 400,
	'rm_sync_freq': 1,
	'rm_algo': 'atomic', # none, u, atomic, conjunctive
}

CONFIG['drawer-env']['rl_train']['lift_red_box'] = {
	'epochs': 10,
	'batch_size': 8000,
	'frames_per_proc': 4000,
	'discount': 0.99,
	'gae_lambda': 0.99,
	'potential_scale': 0.1,
	'lr': 0.0003,
	'entropy_coef': 0.03,
}

CONFIG['drawer-env']['rl_train']['pickup_each_box'] = {
	'epochs': 10,
	'batch_size': 8000,
	'frames_per_proc': 4000,
	'discount': 0.99,
	'gae_lambda': 0.99,
	'potential_scale': 1.,
	'lr': 0.0003,
	'entropy_coef': 0.01,
}

CONFIG['drawer-env']['rl_train']['show_green'] = {
	'epochs': 10,
	'batch_size': 8000,
	'frames_per_proc': 4000,
	'discount': 0.99,
	'gae_lambda': 0.99,
	'potential_scale': 1.,
	'lr': 0.0003,
	'entropy_coef': 0.03,
}

