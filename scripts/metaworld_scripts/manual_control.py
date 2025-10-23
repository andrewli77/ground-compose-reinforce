"""
    This script lets you interact with MetaWorld environments by manually controlling the gripper. The controls are:

    W: Move gripper away
    A: Move gripper left
    S: Move gripper closer
    D: Move gripper right
    Z: Move gripper down
    X: Move gripper up
    C: Tighten gripper
    V: Loosen gripper

    ESC: Quit
    [: Discard current trajectory
"""

import argparse, time, glfw, os, io
import torch, PIL

import numpy as np, gymnasium as gym, torch.nn.functional as F
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer

from config import CONFIG, SETTINGS
from src.envs import rm_env_constructor
from src.vlm_models import get_offline_foundation_model


class PlayWrapper(gym.Wrapper):
    """
        Wrapper to help with MuJoCo rendering. 
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.viewer_set = False
        self.key_pressed = []

    def render(self, mode='human'):
        if not self.viewer_set:
            self.viewer_set = True
            # Turn all the geom groups on
            self.env.unwrapped.mujoco_renderer._viewers['human'] = PlayWrapper.PlayViewer(self.env.unwrapped.model, self.env.unwrapped.data)
            self.env.unwrapped.mujoco_renderer._viewers['human'].cam.fixedcamid = -1
            self.env.unwrapped.mujoco_renderer._viewers['human'].cam.type = 0

        self.env.unwrapped.render_mode = mode
        self.env.unwrapped.render()


    def wrap_obs(self, obs):
        if self.viewer_set:
            self.key_pressed = self.env.unwrapped.mujoco_renderer._viewers['human'].consume_key()
        return obs

    def reset(self):
        obs, info = self.env.reset()
        return self.wrap_obs(obs), info

    def step(self, action):
        next_obs, original_reward, env_terminated, env_truncated, info = self.env.step(action)
        return self.wrap_obs(next_obs), original_reward, env_terminated, env_truncated, info

    class PlayViewer(WindowViewer):
        def __init__(self, model, data):
            super().__init__(model, data)
            self.key_pressed = []
            self.custom_text = None
            glfw.set_window_size(self.window, 840, 680)
            glfw.set_key_callback(self.window, self.key_callback)

        def show_text(self, text):
            self.custom_text = text

        def consume_key(self):
            ret = self.key_pressed
            return ret

        def key_callback(self, window, key, scancode, action, mods):
            if action == glfw.PRESS:
                self.key_pressed.append(key)
            elif action == glfw.RELEASE:
                self.key_pressed.remove(key)

            #super()._key_callback(window, key, scancode, action, mods)

        def _create_full_overlay(self):
            if (self.custom_text):
                self.add_overlay(const.GRID_TOPRIGHT, "Instruction", self.custom_text)
            step = round(self.sim.data.time / self.sim.model.opt.timestep)
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Step", str(step))
            self.add_overlay(const.GRID_BOTTOMRIGHT, "timestep", "%.5f" % self.sim.model.opt.timestep)

class PlayAgent(object):
    """
        Control the MetaWorld agents using keyboard inputs.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def get_action(self, obs):
        keys = self.env.key_pressed
        info = 0

        current = np.array([0, 0, 0, 0])

        if(glfw.KEY_D in keys):
            current += np.array([1, 0, 0, 0])
        if(glfw.KEY_A in keys):
            current += np.array([-1, 0, 0, 0])
        if(glfw.KEY_W in keys):
            current += np.array([0, 1, 0, 0])
        if(glfw.KEY_S in keys):
            current += np.array([0, -1, 0, 0])
        if(glfw.KEY_Z in keys):
            current += np.array([0, 0, 1, 0])
        if(glfw.KEY_X in keys):
            current += np.array([0, 0, -1, 0])
        if(glfw.KEY_C in keys):
            current += np.array([0, 0, 0, 1])
        if(glfw.KEY_V in keys):
            current += np.array([0, 0, 0, -1])
        if (glfw.KEY_ESCAPE in keys):
            info = glfw.KEY_ESCAPE
        if (glfw.KEY_LEFT_BRACKET in keys):
            info = glfw.KEY_LEFT_BRACKET
        if (glfw.KEY_RIGHT_BRACKET in keys):
            info = glfw.KEY_RIGHT_BRACKET

        return current, info

def prettyprint(x):
    output = ""
    for each in x:
        output += "%.2f "%(each)
    return output

if __name__ == '__main__':
    # You can set these display options
    PRINT_GROUND_TRUTH_LABELS=0
    PRINT_PREDICTED_LABELS=0
    PRINT_PREDICTED_PROGRESS=0
    PRINT_PREDICTED_RM_INFO=1

    # Load config from config.py
    assert SETTINGS['env'] == 'drawer-env'
    TASK = SETTINGS['rm_file']
    gamma = CONFIG['drawer-env']['rl_train']['rm_discount']
    average_transition_length = CONFIG['drawer-env']['rl_train']['average_transition_length']
    rm_algo = CONFIG['drawer-env']['rl_train']['rm_algo']

    foundation_model = get_offline_foundation_model("drawer-env", load_weights=True)

    env = PlayWrapper(rm_env_constructor('drawer-env', TASK, gamma, average_transition_length, rm_algo)())
    env.unwrapped.max_path_length = env.unwrapped.time_limit = 10000000000

    # Add manual control
    agent = PlayAgent(env)
    o, info = env.reset()
    classifications, progresses = foundation_model.query(torch.tensor(o['observation'], dtype=torch.float32).unsqueeze(0))
    env.env.initialize_potential(classifications[0], progresses[0])

    while True:
        env.render()
        a, keyboard_info = agent.get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)

        classifications, progresses = foundation_model.query(torch.tensor(o['observation'], dtype=torch.float32).unsqueeze(0))
        rm_reward, potential_reward, rm_done = env.env.sync(classifications[0], progresses[0])

        o, r, term, trunc, new_info = env.step(a)

        if PRINT_GROUND_TRUTH_LABELS:
            print("Ground truth labels:", new_info['labels'])
        if PRINT_PREDICTED_LABELS:
            print("Predicted labels:", prettyprint(classifications[0]))
        if PRINT_PREDICTED_PROGRESS:
            print("Predicted progress:", prettyprint(progresses[0]))
        if PRINT_PREDICTED_RM_INFO:
            print("Reward machine state:", o['rm_state'], "RM potential:", torch.tensor(env.env.current_potential).item())

        info = new_info

        # Quit
        if keyboard_info == glfw.KEY_ESCAPE:
            print("ESC - Quitting")
            break

        # Discard current trajectory and reset
        elif keyboard_info == glfw.KEY_LEFT_BRACKET:
            print("[ - Discarding current trajectory")
            trunc = True

        if (term or trunc):
            o, info = env.reset()
