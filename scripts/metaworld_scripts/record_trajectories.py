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

    ESC: Quit (and save current trajectories to disk)
    [: Discard current trajectory
    ]: Save current trajectory to RAM
"""

from scripts.metaworld_scripts.manual_control import *
from src.envs import metaworld_env_constructor
import time

env = PlayWrapper(metaworld_env_constructor('drawer-open-v2-goal-observable', initialize_random_positions=True))
env.env.max_path_length = 10000000000


# Add manual control
agent = PlayAgent(env)
o, info = env.reset()

obss = []
actions = []
labels = []

dir_name = os.path.join("storage", "drawer-env", "offline_rl")
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if os.path.isfile(os.path.join(dir_name, "obss.npz")):
    obss_np = np.load(os.path.join(dir_name, "obss.npz"))
    obss = [obss_np[k] for k in obss_np]
    actions_np = np.load(os.path.join(dir_name, "actions.npz"))
    actions = [actions_np[k] for k in actions_np]
    labels_np = np.load(os.path.join(dir_name, "labels.npz"))
    labels = [labels_np[k] for k in labels_np]
    print(f"Loaded {len(obss)} trajectories from disk.")
    assert len(obss) == len(actions) == len(labels)

cur_obss = []
cur_actions = []
cur_labels = []

while True:
    env.render()
    a, keyboard_info = agent.get_action(o)
    a = np.clip(a, env.action_space.low, env.action_space.high)

    cur_obss.append(o)
    cur_actions.append(a)
    cur_labels.append(info['labels'])
    print(info['labels'])

    # print(info)

    o, r, term, trunc, info = env.step(a)

    # Quit
    if keyboard_info == glfw.KEY_ESCAPE:
        print("ESC - Quitting")
        break

    # Discard current trajectory and continue
    elif keyboard_info == glfw.KEY_LEFT_BRACKET:
        print("[ - Discarding current trajectory")
        for i in range(10):
            print(".")
        trunc = True

    # Save current trajectory and continue
    elif keyboard_info == glfw.KEY_RIGHT_BRACKET:
        cur_obss.append(o)

        obss.append(np.array(cur_obss, dtype=np.float32))
        actions.append(np.array(cur_actions, dtype=np.float32))
        labels.append(np.array(cur_labels, dtype=np.uint8))


        print(f"] - Saving current trajectory of length {len(cur_actions)}.")
        for i in range(10):
            print(".")
        trunc = True
        time.sleep(0.2)

    if (term or trunc):
        o, info = env.reset()
        print(info)
        cur_obss = []
        cur_actions = []
        cur_labels = []

print(f"Saving {len(obss)} trajectories to disk.")        
np.savez_compressed(os.path.join(dir_name, "obss.npz"), *obss)
np.savez_compressed(os.path.join(dir_name, "actions.npz"), *actions)
np.savez_compressed(os.path.join(dir_name, "labels.npz"), *labels)
