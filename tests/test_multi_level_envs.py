import numpy as np
import pytest

from stable_baselines3.common.envs.multi_level_model.ressim import Grid
from stable_baselines3.common.envs.multi_level_ressim_env import RessimParams, RessimEnvParamGenerator, MultiLevelRessimEnv
from stable_baselines3.common.envs.multi_level_model.generate_env_params import generate_env_case_1_params, generate_env_case_2_params

@pytest.mark.parametrize("dim", [100, 75, 50])
def test_multi_level_env_params(dim):
    """
    check the mulilevel environment parameters
    """
    level_dict = {1:[int(0.25*dim), int(0.25*dim*0.5)], 2:[int(0.5*dim), int(0.5*dim*0.5)], 3:[int(0.75*dim), int(0.75*dim*0.5)], 4:[int(1.0*dim), int(1.0*dim*0.5)]}
    grid = Grid(nx=dim, ny=int(dim*0.5), lx=1, ly=0.5)
    phi = np.ones(grid.shape)
    k = np.array([np.ones(grid.shape)])
    s_wir, s_oir = 0.2, 0.2
    mu_w, mu_o = 0.01, 0.1
    mobility = "linear"
    dt, nstep, terminal_step = 0.1, 5, 5
    q = np.ones(grid.shape)
    s = np.zeros(grid.shape)

    ressim_params = RessimParams(grid, k, phi, s_wir, s_oir, 
                                 mu_w, mu_o, mobility, 
                                 dt, nstep, terminal_step, 
                                 q, s, level_dict)

    ressim_env_params_generator = RessimEnvParamGenerator(ressim_params)

    ressim_params_array = []
    for level in level_dict.keys():
        ressim_params_array.append(ressim_env_params_generator.get_level_env_params(level))

    for ressim_params_l in ressim_params_array:
        assert ressim_params_l.k_list[0].shape == ressim_params_l.grid.shape
        assert ressim_params_l.phi.shape == ressim_params_l.grid.shape
        assert ressim_params_l.q.shape == ressim_params_l.grid.shape
        assert ressim_params_l.s.shape == ressim_params_l.grid.shape

@pytest.mark.parametrize("params_func, level", [(generate_env_case_1_params, 1),
                                                (generate_env_case_1_params, 2),
                                                (generate_env_case_1_params, 3),
                                                (generate_env_case_1_params, 4),
                                                (generate_env_case_2_params, 1),
                                                (generate_env_case_2_params, 2),
                                                (generate_env_case_2_params, 3)])
def test_multiple_grid_envs(params_func, level):
    """
    test transitions of environment at multiple levels
    """
    params_input = params_func()
    params_generator = RessimEnvParamGenerator(params_input)
    params = params_generator.get_level_env_params(level)
    env = MultiLevelRessimEnv(params, level)

    for _ in range(100):
        _, done = env.reset(), False
        while not done:
            assert env.k_load.shape == (env.ressim_params.level_dict[level][1], env.ressim_params.level_dict[level][0])
            assert env.env_action.shape == (env.ressim_params.level_dict[level][1], env.ressim_params.level_dict[level][0])
            assert env.state['s'].shape == (env.ressim_params.level_dict[level][1], env.ressim_params.level_dict[level][0])
            _, _, done, _ = env.step(env.action_space.sample())


@pytest.mark.parametrize("params_func", [generate_env_case_1_params,generate_env_case_2_params])
def test_multi_level_env_mapping(params_func):
    """
    test mapping of environement from different levels during the transitions
    """
    params_input = params_func()
    params_generator = RessimEnvParamGenerator(params_input)

    env_dict = {}
    for level in params_input.level_dict.keys():
        params = params_generator.get_level_env_params(level)
        env_dict[level] = MultiLevelRessimEnv(params, level)

    level = 1
    bump = +1
    L = len( params_input.level_dict )

    for level in params_input.level_dict.keys():
        _ = env_dict[level].reset()
        _ = env_dict[level].reset()

    done=False

    for _ in range(10):
        
        if done:
            print('\nreset')
            _ = env_dict[level].reset()
            # _ = env_dict[level+bump].reset()

        if (level==1 and bump==-1) or (level==L and bump==1):
            bump = -bump

        env_dict[level+bump].map_from(env_dict[level])

        assert env_dict[level].episode_step == env_dict[level+bump].episode_step
        assert env_dict[level].state['s'].shape == (params_input.level_dict[level][1],params_input.level_dict[level][0])
        assert env_dict[level+bump].state['s'].shape == (params_input.level_dict[level+bump][1],params_input.level_dict[level+bump][0])

        action = env_dict[level].action_space.sample() 
        _, r, done, _ = env_dict[level].step(action)
        _, r_, _, _ = env_dict[level+bump].step(action)

        # print(f'({env_dict[level].state['s'].shape} -> {env_dict[level+bump].state['s'].shape})')
        print(f'level({level} -> {level+bump}): reward ({ round(r*100, 2) } -> { round(r_*100, 2) })') 

        level = level + bump


@pytest.mark.parametrize("params_func, level", [(generate_env_case_1_params, 1),
                                                (generate_env_case_1_params, 2),
                                                (generate_env_case_1_params, 3),
                                                (generate_env_case_1_params, 4),
                                                (generate_env_case_2_params, 1),
                                                (generate_env_case_2_params, 2),
                                                (generate_env_case_2_params, 3)])
def test_phi_a_inverse_function(params_func, level):
    """
    test transitions of environment at multiple levels
    """
    params_input = params_func()
    params_generator = RessimEnvParamGenerator(params_input)
    params = params_generator.get_level_env_params(level)
    env = MultiLevelRessimEnv(params, level)

    for _ in range(1):
        _, done = env.reset(), False
        while not done:
            assert env.k_load.shape == (env.ressim_params.level_dict[level][1], env.ressim_params.level_dict[level][0])
            assert env.env_action.shape == (env.ressim_params.level_dict[level][1], env.ressim_params.level_dict[level][0])
            assert env.state['s'].shape == (env.ressim_params.level_dict[level][1], env.ressim_params.level_dict[level][0])
            action = env.action_space.sample()
            q_L = env.phi_a(action)
            action_ = env.phi_a_inverse(q_L)
            q_L_ = env.phi_a(action_)

            assert np.abs(q_L-q_L_).max() < 1e-10, np.abs(q_L-q_L_) 

            _, _, done, _ = env.step(action)

