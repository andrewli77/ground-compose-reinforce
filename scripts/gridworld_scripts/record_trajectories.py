from src.envs.gridworld import *

def generate_random_trajectories(data_dir, n_trajectories):
    env = GridWorldEnv()

    obss = np.empty((n_trajectories, env.time_limit+1, *env.observation_space.shape), dtype=np.float32)
    actions = np.empty((n_trajectories, env.time_limit, *env.action_space.shape), dtype=np.uint8)
    labels = np.empty((n_trajectories, env.time_limit+1, 5), dtype=np.uint8)

    for i in range(n_trajectories):
        obs, info = env.reset()

        for step in range(env.time_limit):
            obss[i, step] = obs
            labels[i, step] = info['labels']

            action = env.action_space.sample()
            actions[i, step] = action

            obs, _, _, _, info = env.step(action)

        obss[i, env.time_limit] = obs
        labels[i, env.time_limit] = info['labels']

    import os
    np.savez_compressed(os.path.join(data_dir, "obss.npz"), obss)
    np.savez_compressed(os.path.join(data_dir, "actions.npz"), actions)
    np.savez_compressed(os.path.join(data_dir, "labels.npz"), labels)


if __name__ == '__main__':
    generate_random_trajectories('storage/grid-env/offline_rl/', 5000)