from src.envs.gridworld import GridWorldEnv
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from src.reward_machines.rm_env import RewardMachineEnv

def rm_env_constructor(env_name, rm_file, gamma, average_transition_length, rm_algo):
	if env_name == 'grid-env':
		return lambda: RewardMachineEnv(GridWorldEnv(), rm_file, gamma, average_transition_length, rm_algo)
	elif env_name == 'drawer-env':
		randomize_metaworld = (rm_file.split("/")[-1] == 'show_green.txt')
		return lambda : RewardMachineEnv(metaworld_env_constructor('drawer-open-v2-goal-observable', randomize_metaworld), rm_file, gamma, average_transition_length, rm_algo)
	else:
		raise NotImplementedError

def env_constructor(env_name):
	if env_name == 'grid-env':
		return lambda: GridWorldEnv()
	elif env_name == 'drawer-env':
		return lambda : metaworld_env_constructor('drawer-open-v2-goal-observable', False)
	else:
		raise NotImplementedError

def metaworld_env_constructor(env_name, initialize_random_positions=False):
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode='rgb_array', initialize_random_positions=initialize_random_positions)
	env.unwrapped.mujoco_renderer.camera_id = 2
	return env