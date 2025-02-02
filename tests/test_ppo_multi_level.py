from curses import wrapper
import pytest
import numpy as np
import os

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

@pytest.mark.parametrize( "tmp_path, n_steps_dict, batch_size_dict , wrapper, generate_params", 
                          [("case_1", {1:30, 2:20, 3:10, 4:5}, {1:30, 2:20, 3:10, 4:5}, SubprocVecMultiLevelEnv, generate_env_case_1_params),
                           ("case_2", {1:15, 2:10, 3:5},{1:15, 2:10, 3:5}, SubprocVecMultiLevelEnv, generate_env_case_2_params)] )
def test_functional_ppo_ml(tmp_path, n_steps_dict, batch_size_dict, wrapper, generate_params):
    # print(get_envs)
    
    params_input = generate_params()
    params_generator = RessimEnvParamGenerator(params_input)

    env_dict = {}
    iter=2
    n_actor = 2
    for level in params_input.level_dict.keys():
        params = params_generator.get_level_env_params(level)
        env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
                                        n_envs=n_actor, 
                                        seed=level, 
                                        env_kwargs= {"ressim_params":params, "level":level}, 
                                        vec_env_cls=wrapper )

    kwargs = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, n_epochs=1, seed=1, device='cpu', target_kl=1e-3)
    model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs)
    model_ppo_ml.learn(sum(n_steps_dict.values())*n_actor*iter)

    # Test model saving
    model_ppo_ml.save(tmp_path+"/test_save.zip", exclude=['env_dict'])
    os.system("rm -rf "+tmp_path)

@pytest.mark.parametrize( "n_steps_dict, batch_size_dict , wrapper, generate_params", 
                          [({1:4}, {1:4}, DummyVecMultiLevelEnv, generate_env_case_1_params ),
                          ({1:4}, {1:4}, DummyVecMultiLevelEnv, generate_env_case_2_params )] )
def test_benchmark_ppo_ml(n_steps_dict, batch_size_dict, wrapper, generate_params):
    # print(get_envs)
    
    params_input = generate_params()
    params_input.level_dict = { 1 : params_input.level_dict[len( params_input.level_dict )] }
    params_generator = RessimEnvParamGenerator(params_input)

    env_dict = {}
    n_cpu = 4
    iter = 2
    for level in params_input.level_dict.keys():
        params = params_generator.get_level_env_params(level)
        env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
                                        n_envs=n_cpu, 
                                        seed=level, 
                                        env_kwargs= {"ressim_params":params, "level":level}, 
                                        vec_env_cls=wrapper )

    kwargs = dict(n_steps=n_steps_dict[1], batch_size=batch_size_dict[1], n_epochs=1, seed=1, device='cpu')
    kwargs_ml = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, n_epochs=1, seed=1, device='cpu')

    model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs_ml)
    model_ppo_ml.learn(sum(n_steps_dict.values())*4*2)

    model_ppo = PPO("MlpPolicy", env_dict[1], verbose=True, **kwargs)
    model_ppo.learn(sum(n_steps_dict.values())*n_cpu*iter)

    return_array, return_ml_array = [], []
    for i in range(3):
        env_dict[1].seed(i)
        return_array.append(evaluate_policy(model_ppo, env_dict[1] ))
        env_dict[1].seed(i)
        return_ml_array.append(evaluate_policy(model_ppo_ml, env_dict[1] ))
    assert np.array_equal(return_array, return_ml_array), print(return_array, '\n', return_ml_array)

@pytest.mark.parametrize( "n_steps_dict, wrapper, generate_params, n_exp, comp_time", 
                          [({ 1:20, 2:10, 3:5}, SubprocVecMultiLevelEnv, generate_env_case_1_params, 3600, {1:0.033,2:0.046,3:0.104}),
                           ({1:20, 2:10, 3:5}, SubprocVecMultiLevelEnv, generate_env_case_2_params, 3600, {1:0.028,2:0.036,3:0.08})] )
def test_functional_ppo_ml_analysis_case_2(n_steps_dict, wrapper, generate_params, n_exp, tmp_path, comp_time):
    # print(get_envs)

    params_input = generate_params()
    fine_level = len(params_input.level_dict.keys())
    n_level = len(n_steps_dict.keys())
    level_dict_ = {}
    for i,l in enumerate(range(fine_level-n_level, fine_level)):
        print(i+1,'->',l+1)
        level_dict_[i+1] = params_input.level_dict[l+1]
    params_input.level_dict = level_dict_
    params_generator = RessimEnvParamGenerator(params_input)

    env_dict = {}
    batch_size_dict = {}
    num_cpu = 8
    for level in params_input.level_dict.keys():
        params = params_generator.get_level_env_params(level)
        env_dict[level] = make_vec_env( MultiLevelRessimEnv, 
                                        n_envs=num_cpu, 
                                        seed=level, 
                                        env_kwargs= {"ressim_params":params, "level":level}, 
                                        vec_env_cls=wrapper )
        batch_size_dict[level] = n_steps_dict[level]*num_cpu

    kwargs = dict(n_steps=n_steps_dict, batch_size=batch_size_dict, n_epochs=5, seed=1, device='cpu', target_kl=1e-1)

    iter=4
    fine_level = len(n_steps_dict)
    model_ppo_ml = PPO_ML("MlpPolicy", env_dict, verbose=True, **kwargs)
    model_ppo_ml.mlmc_analysis(total_timesteps= n_steps_dict[fine_level]*num_cpu*iter , 
                               n_expt=n_exp, 
                               eps_array=[1e-1, 5e-2],
                               analysis_interval=2,
                               step_comp_time_dict=comp_time)
