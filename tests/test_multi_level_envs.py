import numpy as np
import pytest

from stable_baselines3.common.envs.multi_level_model.ressim import Grid
from stable_baselines3.common.envs.multi_level_ressim_env import RessimParams, RessimEnvParamGenerator, MultiLevelRessimEnv
from stable_baselines3.common.envs.multi_level_model.k_distributions import batch_generate, batch_generate_krige

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

def generate_env_case_1_params():

    grid = Grid(nx=61, ny=61, lx=1200*0.3048, ly=1200*0.3048)
    phi = 0.2*np.ones(grid.shape)
    s_wir, s_oir = 0.0, 0.0
    mu_w, mu_o = 3e-4, 3e-4
    mobility = "linear"
    dt, nstep, terminal_step = 5, 5, 5
    ooip = grid.lx * grid.ly * phi[0,0] * (1 - s_wir - s_oir) # original oil in place
    total_time = nstep*terminal_step*dt
    Q = ooip/total_time 
    q = np.zeros(grid.shape)
    q[::2,0] = Q/round(grid.nx/2)
    q[::2,-1] = -Q/round(grid.nx/2)
    s = np.ones(grid.shape)*s_wir

    low_log_k = -2
    high_log_k = 5.5

    # generate random training and evaluation samples
    k = batch_generate(nx=grid.nx, ny=grid.ny, lx=grid.lx, ly=grid.ly, 
                       channel_k=high_log_k, base_k=low_log_k, channel_width_range=(0.1,0.3), 
                       sample_size=9, seed=1)
    md_m2_conv = 1/1.01325e+15
    k = md_m2_conv*np.exp(k)
    level_dict = {1:[8,8], 2:[15,15], 3:[30, 30], 4:[61,61]}

    ressim_params = RessimParams(grid, k, phi, s_wir, s_oir, 
                                 mu_w, mu_o, mobility, 
                                 dt, nstep, terminal_step, 
                                 q, s, level_dict)

    return ressim_params

def generate_env_case_2_params():

    grid = Grid(nx=31, ny=91, lx=620*0.3048, ly=1820*0.3048)
    phi = 0.2*np.ones(grid.shape)
    s_wir, s_oir = 0.0, 0.0
    mu_w, mu_o = 3e-4, 3e-4
    mobility = "linear"
    dt, nstep, terminal_step = 1, 5, 5
    ooip = grid.lx * grid.ly * phi[0,0] * (1 - s_wir - s_oir) # original oil in place
    total_time = nstep*terminal_step*dt
    Q = ooip/total_time 
    q = np.zeros(grid.shape)
    q[::2,0] = Q/round(grid.nx/2)
    q[::2,-1] = -Q/round(grid.nx/2)
    s = np.ones(grid.shape)*s_wir

    tarbert_log_mean = 2.41
    step_x, step_y = grid.lx/(grid.nx-1), grid.ly/(grid.ny-1)
    cond_pos_ = np.where(np.abs(q) > 0.0) 
    cond_pos = np.array([cond_pos_[0]*step_y, cond_pos_[1]*step_x])
    n_wells = cond_pos[0].shape[0]
    cond_val = [tarbert_log_mean]*n_wells

    # generate random training and evaluation samples
    k = batch_generate_krige(nx=grid.nx, ny=grid.ny, lx=grid.lx, ly=grid.ly,
                             variance=5,
                             len_scale=[0.1*grid.lx, 1.0*grid.lx],
                             cond_pos=cond_pos,
                             cond_val=cond_val,
                             angle=np.pi/8,
                             n_samples=16,
                             seed=1)
    md_m2_conv = 1/1.01325e+15
    k = md_m2_conv*np.exp(k)
    level_dict = {1:[7,22], 2:[15, 45], 3:[31,91]}

    ressim_params = RessimParams(grid, k, phi, s_wir, s_oir, 
                                 mu_w, mu_o, mobility, 
                                 dt, nstep, terminal_step, 
                                 q, s, level_dict)

    return ressim_params

@pytest.mark.parametrize("params_func, level", [(generate_env_case_1_params, 1),
                                                (generate_env_case_1_params, 2),
                                                (generate_env_case_1_params, 3),
                                                (generate_env_case_1_params, 4),
                                                (generate_env_case_2_params, 1),
                                                (generate_env_case_2_params, 2),
                                                (generate_env_case_2_params, 3)])
def test_fine_grid_envs(params_func, level):
    params_input = params_func()
    params_generator = RessimEnvParamGenerator(params_input)
    params = params_generator.get_level_env_params(level)
    env = MultiLevelRessimEnv(params, level)

    for _ in range(100):
        _, done = env.reset(), False
        assert env.k_load.shape == env.ressim_params.grid.shape
        assert env.q_load.shape == env.ressim_params.grid.shape
        assert env.s_load.shape == env.ressim_params.grid.shape
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
