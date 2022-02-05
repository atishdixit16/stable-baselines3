import gym
from numpy import indices

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from typing import Any, Callable, List

class SubprocVecMultiLevelEnv(SubprocVecEnv):
    """
    Creates a simple vectorized wrapper for multi level PPO implementation which contains parallel implementation of `map_from` function

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super(SubprocVecMultiLevelEnv, self).__init__(env_fns=env_fns)

    def env_method(self, envs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        indices = self._get_indices(None)
        return [getattr(self.envs[i], 'map_from')(envs[i]) for i in indices]


    def map_from(self, envs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        indices = self._get_indices(None)
        for i in indices:
            self.remote[i].send(("env_method", ('map_from', envs[i])))
        target_remotes = [self.remotes[i] for i in indices]
        return [remote.recv() for remote in target_remotes]