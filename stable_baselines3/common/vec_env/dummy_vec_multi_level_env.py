import gym

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

    def env_method(self, envs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        indices = self._get_indices(None)
        return [getattr(self.envs[i], 'map_from')(envs[i]) for i in indices]
