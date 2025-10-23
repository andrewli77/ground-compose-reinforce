"""
These are simple wrappers that will include RMs to any given environment.
It also keeps track of the RM state as the agent interacts with the envirionment.

However, each environment must implement the following function:
	- *get_events(...)*: Returns the propositions that currently hold on the environment.

Notes:
	- The episode ends if the RM reaches a terminal state or the environment reaches a terminal state.
	- The agent only gets the reward given by the RM.
	- Rewards coming from the environment are ignored.

Extra notes:
This file was originally from 
https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/rm_environment.py
"""

import gymnasium
from gymnasium import spaces
import numpy as np
from src.reward_machines import RewardMachine

class RewardMachineEnv(gymnasium.Wrapper):
	def __init__(self, env, rm_file, gamma=0.99, average_transition_length=5, rm_algo="none"):
		"""
		RM environment
		--------------------
		It adds an RM to the environment:
			- This code keeps track of the current state on the current RM task
			- The id of the RM state is appended to the observations
			- The reward given to the agent comes from the RM

		Parameters
		--------------------
			- env: original environment. It must implement the following function:
			- rm_file: string with path to the RM file.
		"""
		super().__init__(env)

		# Loading the reward machines
		self.rm_file = rm_file

		# Agent reward machine is modelled by the agent, and updated by a (potentially noisy) labelling function.
		# The agent receives and optimizes rewards from it, and also uses its state for decision-making.
		self.agent_reward_machine = RewardMachine(rm_file, rm_algo)
		self.agent_reward_machine.add_reward_shaping(gamma, average_transition_length)

		# True reward machine uses the ground-truth labelling function.
		# IMPORTANT: It should only be used for the purposes of evaluation, not training.
		self.true_reward_machine = RewardMachine(rm_file, 'none')

		self.num_rm_states = len(self.agent_reward_machine.get_states())
		self.gamma = gamma
		
		self.time_limit = self.env.time_limit

		# The observation space is a dictionary including the env features and a one-hot representation of the state in the reward machine
		self.observation_space = spaces.Dict(
			{
				'observation': env.observation_space,
				'rm_state': spaces.Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8),
			})

	def reset(self):
		self.obs, info = self.env.reset()
		self.current_agent_u_id  = self.agent_reward_machine.reset()
		self.current_potential = None

		self.current_true_u_id = self.true_reward_machine.reset()
		return self.get_observation(self.obs, self.current_agent_u_id), info

	def step(self, action):
		next_obs, _, env_terminated, env_truncated, info = self.env.step(action)
		
		# Sync true RM state
		true_rm_rew, _ = self.sync_true_rm(info['labels'])
		info['true_reward'] = true_rm_rew

		return self.get_observation(next_obs, self.current_agent_u_id), 0., env_terminated, env_truncated, info

	def sync_true_rm(self, true_props):
		if self.current_true_u_id == self.true_reward_machine.terminal_u:
			return 0, True
		else:
			self.current_true_u_id, true_rm_rew, true_rm_done = self.true_reward_machine.step(self.current_true_u_id, true_props, None)
			return true_rm_rew, true_rm_done

	# Call this function to update the agent RM state.
	# Returns the reward for the transition and termination info
	# e.g. if propositions are 'a','b','c' you might have
	# true_props: 'ab'
	# dists: {'a': 0.2, '!a': 0.8, 'b': 0.1', '!b': 0.8, 'c': 0.2, '!c': 0.4} 
	def sync(self, true_props, dists):
		self.current_agent_u_id, rm_rew, rm_done = self.agent_reward_machine.step(self.current_agent_u_id, true_props, None)
		potential = self.agent_reward_machine.compute_intrastate_potential(self.current_agent_u_id, true_props, dists)
		assert(self.current_potential is not None)

		if not rm_done:
			potential_reward = potential - self.current_potential
		else:
			potential_reward = 0.
			
		self.current_potential = potential

		return rm_rew, potential_reward, rm_done

	def initialize_potential(self, true_props, dists):
		assert(self.current_potential is None)
		potential = self.agent_reward_machine.compute_intrastate_potential(self.current_agent_u_id, true_props, dists)
		self.current_potential = potential

	def get_observation(self, obs, u_id):
		rm_feat = np.zeros(self.num_rm_states)
		rm_feat[u_id] = 1
		return {'observation': obs,'rm_state': rm_feat}

	def render(self):
		print("RM state:", self.current_u_id, "---", "Potential:", self.current_potential)
		return self.env.render()

	def get_image(self):
		return self.env.get_image()