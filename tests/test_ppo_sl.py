import pytest
import numpy as np
import gym

from stable_baselines3.ppo import PPO
from torch._C import device
from stable_baselines3.ppo_single_level import PPO_SL
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


@pytest.fixture
def get_envs():
    env_ = gym.make("CartPole-v1")
    env_.seed(1)
    envs_ = make_vec_env("CartPole-v1", n_envs=4, seed=1)
    return env_, envs_

@pytest.mark.parametrize("model_class", [PPO])

def test_ppo_sl(get_envs, model_class):
    # print(get_envs)
    env, envs = get_envs[0], get_envs[1]
    kwargs = dict(n_steps=50, batch_size=25, seed=1, device='cpu')

    model = model_class("MlpPolicy", envs, **kwargs).learn(1000)
    model_ppo_sl = PPO_SL("MlpPolicy", envs, **kwargs).learn(1000)

    return_array, return_sl_array = [], []
    for i in range(10):
        env.seed(i)
        return_array.append(evaluate_policy(model, Monitor(env) ))
        env.seed(i)
        return_sl_array.append(evaluate_policy(model_ppo_sl, Monitor(env) ))
    print(return_array, return_sl_array)
    assert np.array_equal(return_array, return_sl_array)
