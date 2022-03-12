from curses import wrapper
import pytest
import numpy as np

from stable_baselines3.ppo import PPO
from stable_baselines3.ppo_multi_level import PPO_ML
from stable_baselines3.ppo_single_level import PPO_SL
from stable_baselines3.common.envs.multi_level_ressim_env import RessimEnvParamGenerator, MultiLevelRessimEnv
from stable_baselines3.common.envs.multi_level_model.generate_env_params import generate_env_case_1_params, generate_env_case_2_params
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_multi_level_env import SubprocVecMultiLevelEnv
from stable_baselines3.common.vec_env.dummy_vec_multi_level_env import DummyVecMultiLevelEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# @pytest.mark.parametrize( "n_steps_dict, batch_size_dict , wrapper, generate_params", 
#                           [({1:30, 2:6, 3:3, 4:1}, {1:120, 2:24, 3:12, 4:4}, DummyVecMultiLevelEnv, generate_env_case_1_params),
#                            ({1:100, 2:50, 3:25},{1:20, 2:10, 3:5}, SubprocVecMultiLevelEnv, generate_env_case_2_params)] )
# def test_functional_ppo_ml(n_steps_dict, batch_size_dict, wrapper, generate_params):
#     # print(get_envs)
    
#     params_input = generate_params()
#     params_generator = RessimEnvParamGenerator(params_input)

#     env_dict = {}
#     for level in params_input.level_dict.keys():
#         params = params_generator.get_level_env_params(level)
#         env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
#                                         n_envs=4, 
#                                         seed=level, 
#                                         env_kwargs= {"ressim_params":params, "level":level}, 
#                                         vec_env_cls=wrapper )

#     kwargs = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, n_epochs=1, seed=1, device='cpu')

#     model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs)
#     model_ppo_ml.learn(sum(n_steps_dict.values())*8)


# @pytest.mark.parametrize( "n_steps_dict, batch_size_dict , wrapper, generate_params", 
#                           [({1:10}, {1:5}, DummyVecMultiLevelEnv, generate_env_case_1_params ),
#                           ({1:10}, {1:5}, DummyVecMultiLevelEnv, generate_env_case_2_params )] )
# def test_benchmark_ppo_ml(n_steps_dict, batch_size_dict, wrapper, generate_params):
#     # print(get_envs)
    
#     params_input = generate_params()
#     params_input.level_dict = { 1 : params_input.level_dict[len( params_input.level_dict )] }
#     params_generator = RessimEnvParamGenerator(params_input)

#     env_dict = {}
#     for level in params_input.level_dict.keys():
#         params = params_generator.get_level_env_params(level)
#         env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
#                                         n_envs=4, 
#                                         seed=level, 
#                                         env_kwargs= {"ressim_params":params, "level":level}, 
#                                         vec_env_cls=wrapper )

#     kwargs = dict(n_steps=n_steps_dict[1], batch_size=batch_size_dict[1], n_epochs=1, seed=1, device='cpu')
#     kwargs_ml = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, n_epochs=1, seed=1, device='cpu')

#     model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs_ml)
#     model_ppo_ml.learn(sum(n_steps_dict.values())*8)

#     model_ppo = PPO("MlpPolicy", env_dict[1], verbose=True, **kwargs)
#     model_ppo.learn(sum(n_steps_dict.values())*8)

#     return_array, return_ml_array = [], []
#     for i in range(10):
#         env_dict[1].seed(i)
#         return_array.append(evaluate_policy(model_ppo, env_dict[1] ))
#         env_dict[1].seed(i)
#         return_ml_array.append(evaluate_policy(model_ppo_ml, env_dict[1] ))
#     print(return_array, '\n', return_ml_array)
#     assert np.array_equal(return_array, return_ml_array), print(return_array, '\n', return_ml_array)

# @pytest.mark.parametrize( "n_steps_dict, wrapper, generate_params", 
#                           [({ 1:3, 2:1}, SubprocVecMultiLevelEnv, generate_env_case_1_params)] )
# def test_functional_ppo_ml_analysis_case_1(n_steps_dict, wrapper, generate_params):
#     # print(get_envs)

#     fine_level = len(n_steps_dict)

#     level_dict = { 1:[15, 15], 2:[61,61]}
    
#     params_input = generate_params()
#     params_input.level_dict = level_dict
#     params_generator = RessimEnvParamGenerator(params_input)

#     env_dict = {}
#     batch_size_dict = {}
#     num_cpu = 4
#     for level in params_input.level_dict.keys():
#         params = params_generator.get_level_env_params(level)
#         env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
#                                         n_envs=num_cpu, 
#                                         seed=level, 
#                                         env_kwargs= {"ressim_params":params, "level":level}, 
#                                         vec_env_cls=wrapper )
#         batch_size_dict[level] = n_steps_dict[level]*num_cpu

#     kwargs = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, n_epochs=1, seed=1, device='cpu')

#     model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs)
#     model_ppo_ml.mlmc_analysis(total_timesteps=batch_size_dict[fine_level]*10, n_expt=300, analysis_interval=2, analysis_batch_size=300)


@pytest.mark.parametrize( "n_steps_dict, wrapper, generate_params", 
                        #   [({1:30, 2:6, 3:3, 4:1}, SubprocVecMultiLevelEnv, generate_env_case_1_params),
                           [({1:4, 2:2, 3:1}, SubprocVecMultiLevelEnv, generate_env_case_2_params)] )
def test_functional_ppo_ml_analysis_case_2(n_steps_dict, wrapper, generate_params):
    # print(get_envs)

    fine_level = len(n_steps_dict)
    
    params_input = generate_params()
    params_generator = RessimEnvParamGenerator(params_input)

    env_dict = {}
    batch_size_dict = {}
    num_cpu = 4
    for level in params_input.level_dict.keys():
        params = params_generator.get_level_env_params(level)
        env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
                                        n_envs=num_cpu, 
                                        seed=level, 
                                        env_kwargs= {"ressim_params":params, "level":level}, 
                                        vec_env_cls=wrapper )
        batch_size_dict[level] = n_steps_dict[level]*num_cpu

    kwargs = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, n_epochs=1, seed=1, device='cpu')

    model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs)
    model_ppo_ml.mlmc_analysis(total_timesteps=batch_size_dict[fine_level]*10, n_expt=300, analysis_interval=2, analysis_batch_size=300)

