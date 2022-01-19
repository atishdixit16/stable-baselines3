import numpy as np
import pytest

from stable_baselines3.common.envs.multi_level_model.ressim import Grid
from stable_baselines3.common.envs.multi_level_ressim_env import RessimParams, RessimEnvParamGenerator

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

    ressim_env_params_generator = RessimEnvParamGenerator(ressim_params, level_dict)

    ressim_params_array = []
    for level in level_dict.keys():
        ressim_params_array.append(ressim_env_params_generator.get_level_env_params(level))

    for ressim_params_l in ressim_params_array:
        assert ressim_params_l.k_list[0].shape == ressim_params_l.grid.shape
        assert ressim_params_l.phi.shape == ressim_params_l.grid.shape
        assert ressim_params_l.q.shape == ressim_params_l.grid.shape
        assert ressim_params_l.s.shape == ressim_params_l.grid.shape


        