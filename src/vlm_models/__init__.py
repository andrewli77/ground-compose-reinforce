from .grid_env_models import *
from .drawer_env_models import *
from src.envs import env_constructor
from config import CONFIG, SETTINGS
import os

def get_foundation_model(env_name, model_type, device=None):
	if env_name == "grid-env":
		if model_type == "ground_truth":
			print("Using ground truth grid-env propositional model.")
			return GridEnvGTFoundationModel()
		elif model_type == "offline_rl":
			print("Using pretrained grid-env propositional model.")
			return get_offline_foundation_model(env_name, device, load_weights=True)

	if env_name == "drawer-env":
		if model_type == "offline_rl":
			print("Using pretrained drawer-env propositional model.")
			return get_offline_foundation_model(env_name, device, load_weights=True)

	raise NotImplementedError

def get_offline_foundation_model(env_name, device=None, load_weights=False):
	model = None

	if env_name == "grid-env":
		model = GridEnvFoundationModel(5, 19)
		classifier_weight_dir = os.path.join(CONFIG['grid-env']['offline_rl']['data_dir'], "classifier.pt")
		progress_weight_dir = os.path.join(CONFIG['grid-env']['offline_rl']['data_dir'], "progress.pt")

	if env_name == "drawer-env":
		model = DrawerEnvFoundationModel(11, 27)
		classifier_weight_dir = os.path.join(CONFIG['drawer-env']['offline_rl']['data_dir'], "classifier.pt")
		progress_weight_dir = os.path.join(CONFIG['drawer-env']['offline_rl']['data_dir'], "progress.pt")

	if model is None:
		raise NotImplementedError	

	if load_weights:
		classifier_state_dict = torch.load(classifier_weight_dir, map_location=device if device else "cpu")['classifier_model']
		progress_state_dict = torch.load(progress_weight_dir, map_location=device if device else "cpu")['progress_model']
		model.classifier.load_state_dict(classifier_state_dict)
		model.progress.load_state_dict(progress_state_dict)
		model.eval()
		model.requires_grad = False
	if device:
		model = model.to(device)
	return model

# def get_classifier_model(env_name, model_type, load_weights=False, device=None):
# 	"""
# 	env_name: ['grid-env']
# 	model_type: ['ground_truth', 'offline_rl']
# 	"""
# 	env = env_constructor(env_name)()
# 	model = None

# 	if env_name == 'grid-env':
# 		if model_type == "ground_truth":
# 			model = GridEnvGTClassifierModel()

# 		elif model_type == "offline_rl":
# 			model = GridCNNClassifierModel(env.observation_space, 5).to(device)
# 			if load_weights:
# 				state_dict = torch.load(os.path.join(CONFIG['grid-env']['offline_rl']['data_dir'], "labelling_function.pt"), map_location=device if device else "cpu")['classifier_model']
# 				model.load_state_dict(state_dict)
# 				model.eval()
# 				model.requires_grad = False
# 			if device:
# 				model = model.to(device)
# 	if model is None:
# 		raise NotImplementedError()

# 	return model

# def get_progress_model(env_name, model_type, rm_algo, load_weights=False, device=None):
# 	"""
# 	env_name: ['grid-env']
# 	model_type: ['ground_truth', 'offline_rl']
# 	rm_algo: ['none', 'u', 'atomic', 'conjunctive']
# 	"""
# 	env = env_constructor(env_name)()
# 	model = None

# 	if env_name == 'grid-env':
# 		if model_type == "ground_truth":
# 			model = GridEnvGTProgressModel(rm_algo)

# 		elif model_type == "offline_rl":
# 			if rm_algo == "atomic":
# 				model = GridCNNProgressModel(env.observation_space, env.action_space, 5).to(device)
# 				weight_dir = os.path.join(CONFIG['grid-env']['offline_rl']['data_dir'], "atomic_vfs.pt")

# 			elif rm_algo == "conjunctive":
# 				model = GridCNNProgressModel(env.observation_space, env.action_space, 2).to(device)
# 				weight_dir = os.path.join(CONFIG['grid-env']['offline_rl']['data_dir'], "conjunctive_vfs.pt")
# 			else:
# 				raise NotImplementedError

# 			if load_weights:
# 				state_dict = torch.load(weight_dir, map_location=device if device else "cpu")['qv_model']
# 				model.load_state_dict(state_dict)
# 				model.eval()
# 				model.requires_grad = False
# 			if device:
# 				model = model.to(device)

# 	return model