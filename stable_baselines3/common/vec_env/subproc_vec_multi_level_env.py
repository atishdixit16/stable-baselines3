import numpy as np
import gym
from matplotlib.pyplot import grid
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

    def get_current_obs(self) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(None)
        for remote in target_remotes:
            remote.send(("get_attr", "obs"))
        return np.array( [remote.recv() for remote in target_remotes] )

    def map_from(self, env, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        grids = env.get_attr('grid')
        levels = env.get_attr('level')
        states = env.get_attr('state')
        k_indices = env.get_attr('k_index')
        episode_steps = env.get_attr('episode_step')

        target_remotes = self._get_target_remotes(None)
        for remote, grid, level, state, k_index, episode_step in zip(target_remotes, grids, levels, states, k_indices, episode_steps):
            remote.send( ("env_method", ("map_from_", (grid, level, state, k_index, episode_step), method_kwargs ) ) )
        return [remote.recv() for remote in target_remotes]
