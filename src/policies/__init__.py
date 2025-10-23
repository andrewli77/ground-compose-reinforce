from .model import *
from src.envs import rm_env_constructor

def get_acmodel(env_name, rm_file):
    env = rm_env_constructor(env_name, rm_file, 0.97, 5, None)()

    if env_name == "grid-env":
        return GridACModel(env.observation_space, env.action_space)
    elif env_name == "drawer-env":
        return DrawerACModel(env.observation_space, env.action_space)
    else:
        raise NotImplementedError