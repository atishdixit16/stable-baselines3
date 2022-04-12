from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from tqdm import trange
from stable_baselines3.common import distributions

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.buffer_multi_level import RolloutBufferMultiLevel
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv


class OnPolicyAlgorithmMultiLevel(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """


    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: 'dict[int: Union[GymEnv, str]]',
        learning_rate: Union[float, Schedule],
        n_steps: 'dict[int: int]',
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OnPolicyAlgorithmMultiLevel, self).__init__(
            policy=policy,
            env=env[1],
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.env_dict = env
        self.n_steps_dict = n_steps

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_dict = {}
        self.sync_rollout_buffer_dict = {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBufferMultiLevel

        for level in self.n_steps_dict.keys():
            self.rollout_buffer_dict[level] = buffer_cls(self.n_steps_dict[level],
                                                         self.observation_space,
                                                         self.action_space,
                                                         device=self.device,
                                                         gamma=self.gamma,
                                                         gae_lambda=self.gae_lambda,
                                                         n_envs=self.n_envs )
            self.sync_rollout_buffer_dict[level] = buffer_cls(self.n_steps_dict[level],
                                                         self.observation_space,
                                                         self.action_space,
                                                         device=self.device,
                                                         gamma=self.gamma,
                                                         gae_lambda=self.gae_lambda,
                                                         n_envs=self.n_envs )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def _setup_analysis(self, num_expt) -> None:

        assert num_expt%self.n_envs ==0, 'please choose `num_expt` to be a factor of `n_envs`'

        self.num_expt = num_expt
        self.analysis_rollout_buffer_dict = {}
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBufferMultiLevel
        buffer_size_per_actor = int( self.num_expt/self.n_envs )
        for level in self.n_steps_dict.keys():
            self.analysis_rollout_buffer_dict[level] = buffer_cls(buffer_size_per_actor,
                                                         self.observation_space,
                                                         self.action_space,
                                                         device=self.device,
                                                         gamma=self.gamma,
                                                         gae_lambda=self.gae_lambda,
                                                         n_envs=self.n_envs )

    def collect_rollouts(
        self,
        env_dict: 'dict[int: VecEnv]',
        callback: BaseCallback,
        rollout_buffer_dict: 'dict[int: RolloutBuffer]',
        sync_rollout_buffer_dict: 'dict[int: RolloutBuffer]',
        n_rollout_steps: 'dict[int: int]',
        ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        for rollout_buffer, sync_rollout_buffer in zip(rollout_buffer_dict.values(), sync_rollout_buffer_dict.values()):
            rollout_buffer.reset()
            sync_rollout_buffer.reset()

        for level in np.sort( list(env_dict.keys()) ):

            if level > 1:
                env_dict[level].map_from((env_dict[level-1]))
            
            n_steps = 0
            # Sample new weights for the state dependent exploration
            if self.use_sde:
                self.policy.reset_noise(env_dict[level].num_envs)
            callback.on_rollout_start()

            while n_steps < n_rollout_steps[level]:

                if level > 1:
                    env_dict[level-1].map_from(env_dict[level])
            
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.policy.reset_noise(env_dict[level].num_envs)

                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy.forward(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env_dict[level].step(clipped_actions)

                self.num_timesteps += env_dict[level].num_envs

                # Give access to local variables
                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

                self._update_info_buffer(infos)
                n_steps += 1

                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # Handle timeout by bootstraping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer_dict[level].add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
                # self._last_obs = new_obs
                # self._last_episode_starts = dones

                if level > 1:
                    # collect rollout in synchronised rollout buffer
                    with th.no_grad():
                        # Convert to pytorch tensor or to TensorDict
                        obs_ = self.env_dict[level-1].get_current_obs()
                        obs_tensor_ = obs_as_tensor(obs_, self.device)
                        actions_ = actions
                        _, values_, _ = self.policy.forward(obs_tensor_)
                        log_probs_ = self.policy.get_distribution(obs_tensor_).log_prob(th.from_numpy(actions_).to(self.device))

                    # Clip the actions to avoid out of bound error
                    if isinstance(self.action_space, gym.spaces.Box):
                        clipped_actions_ = np.clip(actions_, self.action_space.low, self.action_space.high)

                    new_obs_, rewards_, _, infos_ = env_dict[level-1].step(clipped_actions_)

                    self._update_info_buffer(infos_)

                    if isinstance(self.action_space, gym.spaces.Discrete):
                        # Reshape in case of discrete action
                        actions_ = actions_.reshape(-1, 1)

                    # Handle timeout by bootstraping with value function
                    # see GitHub issue #633
                    for idx, done in enumerate(dones):
                        if (
                            done
                            and infos_[idx].get("terminal_observation") is not None
                            and infos_[idx].get("TimeLimit.truncated", False)
                        ):
                            terminal_obs = self.policy.obs_to_tensor(infos_[idx]["terminal_observation"])[0]
                            with th.no_grad():
                                terminal_value = self.policy.predict_values(terminal_obs)[0]
                            rewards_[idx] += self.gamma * terminal_value

                    sync_rollout_buffer_dict[level].add(obs_, actions_, rewards_, self._last_episode_starts, values_, log_probs_)

                    # env_dict[level].reset_from_dones(dones)

                self._last_obs = new_obs
                self._last_episode_starts = dones

            with th.no_grad():
                # Compute value for the last timestep
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            rollout_buffer_dict[level].compute_returns_and_advantage(last_values=values, dones=dones)

            if level > 1:
                with th.no_grad():
                    # Compute value for the last timestep
                    values_ = self.policy.predict_values(obs_as_tensor(new_obs_, self.device))
                sync_rollout_buffer_dict[level].compute_returns_and_advantage(last_values=values_, dones=dones)

        
        callback.on_rollout_end()

        return True


    def collect_analysis_rollouts(
        self,
        env_dict: 'dict[int: VecEnv]',
        callback: BaseCallback,
        analysis_rollout_buffer_dict: 'dict[int: RolloutBuffer]',
        n_rollout_steps: 'int',
        ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        fine_level = len(self.n_steps_dict)

        for analysis_rollout_buffer in analysis_rollout_buffer_dict.values():
            analysis_rollout_buffer.reset()

        
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env_dict[fine_level].num_envs)
        callback.on_rollout_start()

        # To avoid progress bar when not doing mlmc analysis
        if n_rollout_steps==self.n_steps_dict[fine_level]:
            loop_till = range
        else:
            loop_till = trange

        for n_steps in loop_till(n_rollout_steps):

            for level in range(1,fine_level):
                env_dict[level].map_from(env_dict[fine_level])
        
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env_dict[fine_level].num_envs)


            before = time.time()
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            rollout_time = time.time() - before

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env_dict[fine_level].step(clipped_actions)

            self._update_info_buffer(infos)
            # n_steps += 1

            if n_steps <= self.n_steps_dict[fine_level]:
                self.num_timesteps += env_dict[fine_level].num_envs

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            comp_times = np.array([rollout_time/self.n_envs + self.step_comp_time_dict[fine_level]]*self.n_envs)
            analysis_rollout_buffer_dict[fine_level].record_times(comp_times)
            analysis_rollout_buffer_dict[fine_level].add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            last_obs_dict = {}
            last_values_dict = {}

            for level in range(1,fine_level):
                before = time.time()
                # collect rollout in synchronised rollout buffer
                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_ = self.env_dict[level].get_current_obs()
                    obs_tensor_ = obs_as_tensor(obs_, self.device)
                    actions_ = actions
                    _, values_, _ = self.policy.forward(obs_tensor_)
                    log_probs_ = self.policy.get_distribution(obs_tensor_).log_prob(th.from_numpy(actions_).to(self.device))

                rollout_time = time.time() - before

                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions_ = np.clip(actions_, self.action_space.low, self.action_space.high)

                new_obs_, rewards_, _, infos_ = env_dict[level].step(clipped_actions_)

                self._update_info_buffer(infos_)

                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions_ = actions_.reshape(-1, 1)

                # Handle timeout by bootstraping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos_[idx].get("terminal_observation") is not None
                        and infos_[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos_[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards_[idx] += self.gamma * terminal_value

                comp_times = np.array([rollout_time/self.n_envs + self.step_comp_time_dict[level]]*self.n_envs)
                analysis_rollout_buffer_dict[level].record_times(comp_times)
                analysis_rollout_buffer_dict[level].add(obs_, actions_, rewards_, self._last_episode_starts, values_, log_probs_)

                # env_dict[level].reset_from_dones(dones)
                last_obs_dict[level] = new_obs_
                last_values_dict[level] = values_

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        analysis_rollout_buffer_dict[fine_level].compute_returns_and_advantage(last_values=values, dones=dones)

        for level in range(1,fine_level):
            with th.no_grad():
                # Compute value for the last timestep
                values_ = self.policy.predict_values(obs_as_tensor(last_obs_dict[level], self.device))
            analysis_rollout_buffer_dict[level].compute_returns_and_advantage(last_values=last_values_dict[level], dones=dones)

        
        callback.on_rollout_end()

        return True


    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithmMultiLevel",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithmMultiLevel":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        # reset rest of the environemnts (besides level 1, which is done in `_setup_learn`) in the env_dict
        for level in self.env_dict.keys():
            if level > 1:
                _ = self.env_dict[level].reset()

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env_dict, callback, self.rollout_buffer_dict, self.sync_rollout_buffer_dict, self.n_steps_dict)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def mlmc_analysis(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithmMultiLevel",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        n_expt: int = 100,
        eps_array: 'list[float]' = [0.1, 0.05],
        analysis_interval: int = 100,
        analysis_log_path: str = None,
        step_comp_time_dict: 'dict[int: float]'=None
    ) -> "OnPolicyAlgorithmMultiLevel":

        self.iteration = 0
        self.analysis_report = {}
        self.analysis_log_path = analysis_log_path
        self.eps_array = eps_array
        assert step_comp_time_dict is not None, 'provide a dictionary of simulation step time for each level'
        self.step_comp_time_dict = step_comp_time_dict

        fine_level = len(self.env_dict)

        self._setup_analysis(n_expt)
        

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        # reset rest of the environemnts (besides level 1, which is done in `_setup_learn`) in the env_dict
        for level in self.env_dict.keys():
            if level > 1:
                _ = self.env_dict[level].reset()
        
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            self.iteration += 1

            if self.iteration % analysis_interval == 0:
                print('collect rollouts for MLMC analysis...')
                n_rollout_steps = int( self.num_expt/self.n_envs )
            else:
                n_rollout_steps = self.n_steps_dict[fine_level]

            continue_training = self.collect_analysis_rollouts(self.env_dict, callback, self.analysis_rollout_buffer_dict, n_rollout_steps)

            if continue_training is False:
                break

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and self.iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", self.iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train_with_fine_level()

            if self.iteration % analysis_interval == 0:
                print(f'analysis of MLMC estimator for {self.num_expt} number of experimets...')
                self.analysis()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
