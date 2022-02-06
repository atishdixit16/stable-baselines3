from turtle import done
import gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from typing import Any, Callable, List

class DummyVecMultiLevelEnv(DummyVecEnv):
    """
    Creates a simple vectorized wrapper for multi level PPO implementation which contains parallel implementation of `map_from` function

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        super(DummyVecMultiLevelEnv, self).__init__(env_fns=env_fns)

    def map_from(self, env) -> List[Any]:
        """Call instance methods of vectorized environments."""
        indices = self._get_indices(None)
        return [getattr(self.envs[i], 'map_from')(env.envs[i]) for i in indices]

    # def reset_from_dones(self, dones) -> None:
    #     """Call reset from external values (synchronised, in case of multi level) of dones"""
    #     for env_idx, done in zip(range(self.num_envs), dones):
    #         if done:
    #             obs = self.envs[env_idx].reset()
    #             self._save_obs(env_idx, obs)


    def get_current_obs(self) -> VecEnvObs: 
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].obs
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()
