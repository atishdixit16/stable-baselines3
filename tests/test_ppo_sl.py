from tabnanny import verbose
import pytest
import numpy as np
import gym

from stable_baselines3.ppo import PPO
from torch._C import device
from stable_baselines3.ppo_single_level import PPO_SL
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def get_envs(env_str: str):
    env_ = gym.make(env_str)
    env_.seed(1)
    envs_ = make_vec_env(env_str, n_envs=4, seed=1)
    return env_, envs_

@pytest.mark.parametrize("env_str, n_steps_", [("CartPole-v1", 10),
                                               ("Pendulum-v0", 20),
                                               ("Acrobot-v1", 50),
                                               ("MountainCar-v0", 100)])

def test_ppo_sl(env_str, n_steps_):
    # print(get_envs)
    env, envs = get_envs(env_str)
    kwargs = dict(n_steps=n_steps_, batch_size=n_steps_, seed=1, device='cpu')
    kwargs_sl = dict(n_steps={1:n_steps_}, batch_size={1:n_steps_}, seed=1, device='cpu')

    model = PPO("MlpPolicy", envs, **kwargs).learn(n_steps_*16)
    model_ppo_sl = PPO_SL("MlpPolicy", {1:envs}, **kwargs_sl).learn(n_steps_*16)

    return_array, return_sl_array = [], []
    for i in range(10):
        env.seed(i)
        return_array.append(evaluate_policy(model, Monitor(env) ))
        env.seed(i)
        return_sl_array.append(evaluate_policy(model_ppo_sl, Monitor(env) ))
    print(return_array, '\n', return_sl_array)
    assert np.array_equal(return_array, return_sl_array), print(return_array, '\n', return_sl_array)
