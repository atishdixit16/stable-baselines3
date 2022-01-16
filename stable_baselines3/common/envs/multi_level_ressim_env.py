import numpy as np

from numpy import sum, mean
from scipy.stats import hmean

from stable_baselines3.common.envs.multi_level_model.level_mapping_functions import get_accmap, fine_to_coarse_mapping
from stable_baselines3.common.envs.multi_level_model.ressim import Grid


class RessimParams():
    def __init__(self,
                 grid: Grid , k: np.ndarray, phi: np.ndarray, s_wir: float, s_oir: float, # domain properties
                 mu_w: float, mu_o: float, mobility: str,                                 # fluid properties
                 dt: float, nstep: int, terminal_step: int,                               # timesteps
                 q: np.ndarray, s: np.ndarray) -> None:                                   # initial conditions

        """
        reservoir simulation parameters 

        :param grid: 2d grid for the reservoir simulation (an object of class ressim.Grid)
        :param k: a numpy array 2-d fields of permeability samples
        :param phi: reservoir porosity
        :param s_wir: irreducible water saturation in the reservoir
        :param s_oir: irreducible oil saturation in the reservoir 
        :param mu_w: water viscosity
        :param mu_o: oil viscosity
        :param mobility: mobility ratio computation method (linear or quadratic)
        :param dt: simulation timestep
        :param nstep: number simulation timesteps to perform in a single control step
        :param terminal_step: total number control steps
        :param q: 2-d field for source/sink at initial timestep
        :param s: 2-d field for saturation at initial timestep

        """

        self.grid = grid
        self.k = k 
        self.phi = phi
        self.s_wir = s_wir
        self.s_oir = s_oir
        self.mu_w = mu_w
        self.mu_o = mu_o  
        self.mobility = mobility
        self.dt = dt
        self.nstep = nstep
        self.terminal_step = terminal_step
        self.q = q
        self.s = s



class RessimEnvParamGenerator():
    def __init__(self,
                 ressim_params: RessimParams,
                 level_dict: dict) -> None:

        """
        a parameter generator for the MultiLevelRessimEnv environment

        :param ressim_params: reservoir simulation parameters
        :param level_dict: level dictionary

        """

        # check the level_dict input
        for i,l in enumerate(level_dict.keys()):
            assert i+1==l, 'level_dict keys should start from one to the lenth of the dictionary'
            if i>0:
                assert level_dict[l] <= 1 and level_dict[l] > level_dict[l-1], 'level_dict values should reflect grid fidelity factors in ascending order such that the last value is one'

        self.ressim_params = ressim_params
        self.level_dict = level_dict

    def get_level_env_params(self, level: int):
        assert level in self.level_dict.keys(), 'invalid level value, should be among the level_dict keys'
        coarse_grid = Grid(nx=int( self.level_dict[level]*self.ressim_params.grid.nx), 
                           ny=int( self.level_dict[level]*self.ressim_params.grid.ny),
                           lx=self.ressim_params.grid.lx,
                           ly=self.ressim_params.grid.ly)
        accmap = get_accmap(self.ressim_params.grid, coarse_grid)
        coarse_phi = fine_to_coarse_mapping(self.ressim_params.phi, accmap, func=mean)
        coarse_q = fine_to_coarse_mapping(self.ressim_params.q, accmap, func=sum)
        coarse_s = fine_to_coarse_mapping(self.ressim_params.s, accmap, func=mean)
        coarse_k = []
        for k in self.ressim_params.k:
            coarse_k.append(fine_to_coarse_mapping(k, accmap, func=hmean))
        coarse_k = np.array(coarse_k)

        coarse_params = tuple((coarse_grid, 
                               coarse_k, 
                               coarse_phi, 
                               self.ressim_params.s_wir, 
                               self.ressim_params.s_oir, 
                               self.ressim_params.mu_w, 
                               self.ressim_params.mu_o, 
                               self.ressim_params.mobility, 
                               self.ressim_params.dt, 
                               self.ressim_params.nstep, 
                               self.ressim_params.terminal_step, 
                               coarse_q, 
                               coarse_s))

        return RessimParams(*coarse_params)

        

        
