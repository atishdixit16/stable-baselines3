from curses import wrapper
import pytest

from stable_baselines3.ppo_multi_level import PPO_ML
from stable_baselines3.common.envs.multi_level_ressim_env import RessimEnvParamGenerator, MultiLevelRessimEnv
from stable_baselines3.common.envs.multi_level_model.generate_env_params import generate_env_case_1_params, generate_env_case_2_params
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_multi_level_env import SubprocVecMultiLevelEnv
from stable_baselines3.common.vec_env.dummy_vec_multi_level_env import DummyVecMultiLevelEnv

@pytest.mark.parametrize( "n_steps_dict, batch_size_dict , wrapper, generate_params", 
                          [({1:300, 2:60, 3:30, 4:10}, {1:60, 2:12, 3:6, 4:2}, DummyVecMultiLevelEnv, generate_env_case_1_params), 
                           ({1:100, 2:50, 3:25},{1:20, 2:10, 3:5}, SubprocVecMultiLevelEnv, generate_env_case_2_params)] )
def test_functional_ppo_ml(n_steps_dict, batch_size_dict, wrapper, generate_params):
    # print(get_envs)
    
    params_input = generate_params()
    params_generator = RessimEnvParamGenerator(params_input)

    env_dict = {}
    for level in params_input.level_dict.keys():
        params = params_generator.get_level_env_params(level)
        env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
                                        n_envs=4, 
                                        seed=level, 
                                        env_kwargs= {"ressim_params":params, "level":level}, 
                                        vec_env_cls=wrapper )

    kwargs = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, seed=1, device='cpu')

    model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs)
    model_ppo_ml.learn(sum(n_steps_dict.values())*16)