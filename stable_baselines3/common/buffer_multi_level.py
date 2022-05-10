from os import times
from typing import Generator, Optional, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class AnalysisRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    times: th.Tensor

class RolloutBufferMultiLevel(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithm PPO_SL.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBufferMultiLevel, self).__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)
        self.times = None
        self.reset()

    def reset(self) -> None:
        self.times = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super(RolloutBufferMultiLevel, self).reset()

    def record_times(self, comp_times: np.ndarray) -> None:
        'warning: usage only valid if this function is excuted right before `add` function'
        self.times[self.pos] = comp_times

    def swap_and_flatten_for_analysis(self, arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:], order='F')

    def get_analysis_batch(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
            indices = list(range(batch_size))
        else:
            indices = np.random.permutation(self.buffer_size * self.n_envs)
        
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "times",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten_for_analysis(self.__dict__[tensor])
            self.generator_ready = True

        start_idx = 0
        while start_idx < batch_size:
            yield self._get_analysis_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size


    def _get_analysis_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.times[batch_inds].flatten(),
        )
        return AnalysisRolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def get_sync(self, sync_rollout_buffer, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
                sync_rollout_buffer.__dict__[tensor] = sync_rollout_buffer.swap_and_flatten(sync_rollout_buffer.__dict__[tensor])
            self.generator_ready = True
            sync_rollout_buffer.generator_ready = True


        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            yield sync_rollout_buffer._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
