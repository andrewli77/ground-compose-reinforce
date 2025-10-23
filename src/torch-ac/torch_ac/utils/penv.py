import multiprocessing, io, PIL
import gymnasium as gym
import src.envs as envs


multiprocessing.set_start_method("spawn", force=True)

def worker_fn(conn, env_name, rm_file, rm_discount, average_transition_length, rm_algo, seed, i):
    env = envs.rm_env_constructor(env_name, rm_file, rm_discount, average_transition_length, rm_algo)()

    while True:
        cmd, data = conn.recv()

        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            conn.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            obs, info = env.reset()
            conn.send((obs, info))
        elif cmd == "get_image":
            conn.send(env.get_image())
        elif cmd == "sync":
            conn.send(env.sync(*data))
        elif cmd == "initialize_potential":
            env.initialize_potential(*data)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, env_name, rm_file, rm_discount, average_transition_length, rm_algo, seed, n_procs):
        self.locals, self.remotes = zip(*[multiprocessing.Pipe() for _ in range(n_procs-1)])
        self.workers = [
            multiprocessing.Process(target=worker_fn, args=(self.remotes[i], env_name, rm_file, rm_discount, average_transition_length, rm_algo, seed, i+1))
            for i in range(n_procs-1)
        ]

        # Start worker processes
        for worker in self.workers:
            worker.daemon = True
            worker.start()

        # Close the remote ends in the main process
        for remote in self.remotes:
            remote.close()

        self.head_env = envs.rm_env_constructor(env_name, rm_file, rm_discount, average_transition_length, rm_algo)()
        self.observation_space = self.head_env.observation_space
        self.action_space = self.head_env.action_space

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))

        results = zip(*[self.head_env.reset()] + [local.recv() for local in self.locals])
        return results

    def get_image(self):
        for local in self.locals:
            local.send(("get_image", None))
        img = self.head_env.get_image()
        results = [img] + [local.recv() for local in self.locals]
        return tuple(results) 

    def sync(self, true_props, progress_scores):
        for local, clss, dists in zip(self.locals, true_props[1:], progress_scores[1:]):
            local.send(("sync", (clss, dists)))
        
        results = zip(*[self.head_env.sync(true_props[0], progress_scores[0])] + [local.recv() for local in self.locals])
        return results

    def initialize_potential(self, true_props, progress_scores):
        for local, clss, dists in zip(self.locals, true_props[1:], progress_scores[1:]):
            local.send(("initialize_potential", (clss, dists)))

        self.head_env.initialize_potential(true_props[0], progress_scores[0])

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, terminated, truncated, info = self.head_env.step(actions[0])
        results = zip(*[(obs, reward, terminated, truncated, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    """
    Functions with masks. Only applies the given function to the environments where mask is True. The mask must be a tuple of shape
    (n_procs,), and the outputs will also be tuples of shape (n_procs,)
    """
    def reset_by_mask(self, mask):
        results = []

        for i in range(1, len(mask)):
            if mask[i]:
                self.locals[i-1].send(("reset", None))
        
        if mask[0]:
            results.append(self.head_env.reset())

        for i in range(1, len(mask)):
            if mask[i]:
                results.append(self.locals[i-1].recv())
        return zip(*results)

    def get_image_by_mask(self, mask):
        results = []

        for i in range(1, len(mask)):
            if mask[i]:
                self.locals[i-1].send(("get_image", None))
        
        if mask[0]:
            results.append(self.head_env.get_image())

        for i in range(1, len(mask)):
            if mask[i]:
                results.append(self.locals[i-1].recv())
        return tuple(results)

    def initialize_potential_by_mask(self, true_props, progress_scores, mask):
        j = 0

        for i in range(1, len(mask)):
            if mask[i]:
                self.locals[i-1].send(("initialize_potential", (true_props[j], progress_scores[j])))
                j += 1
        
        if mask[0]:
            self.head_env.initialize_potential(true_props[0], progress_scores[0])
