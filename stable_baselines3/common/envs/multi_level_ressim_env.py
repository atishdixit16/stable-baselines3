import numpy as np
import gym
import functools

from numpy import sum, mean
from scipy.stats import hmean
from gym import spaces
from gym.utils import seeding

from stable_baselines3.common.envs.multi_level_model.ressim import SaturationEquation, PressureEquation
from stable_baselines3.common.envs.multi_level_model.utils import linear_mobility, quadratic_mobility, lamb_fn, f_fn, df_fn
from stable_baselines3.common.envs.multi_level_model.level_mapping_functions import coarse_to_fine_mapping, get_accmap, fine_to_coarse_mapping
from stable_baselines3.common.envs.multi_level_model.ressim import Grid

class RessimParams():
    def __init__(self,
                 grid: Grid , k: np.ndarray, phi: np.ndarray, s_wir: float, s_oir: float, # domain properties
                 mu_w: float, mu_o: float, mobility: str,                                 # fluid properties
                 dt: float, nstep: int, terminal_step: int,                               # timesteps
                 q: np.ndarray, s: np.ndarray,                                            # initial conditions
                 level_dict: dict) -> None:

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

        assert k.ndim==3, 'Invalid value k. n permeabilities should be provided as a numpy array with shape (n,grid.nx, grid.ny)'
        assert mobility in ['linear', 'quadratic'], 'invalid mobility parameter. should be one of these: linear, quadratic'

        self.grid = grid
        self.k_list = k 
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
        self.level_dict = level_dict

        # original oil in place
        self.ooip = self.grid.lx * self.grid.ly * self.phi[0,0] * (1 - self.s_wir - self.s_oir)

        self.define_accmap()
        self.define_model_functions()

    def define_model_functions(self):
        # Model functions (mobility and fractional flow function)
        if self.mobility=='linear':
            self.mobi_fn = functools.partial(linear_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        elif self.mobility=='quadratic':
            self.mobi_fn = functools.partial(quadratic_mobility, mu_w=self.mu_w, mu_o=self.mu_o, s_wir=self.s_wir, s_oir=self.s_oir)  # quadratic mobility model
        else:
            raise Exception('invalid mobility input. should be one of these: linear or quadratic')
        self.lamb_fn = functools.partial(lamb_fn, mobi_fn=self.mobi_fn)  # total mobility function
        self.f_fn = functools.partial(f_fn, mobi_fn=self.mobi_fn)  # water fractional flow function
        self.df_fn = functools.partial(df_fn, mobi_fn=self.mobi_fn)

    def define_obs_act_spaces(self, q_fine) -> None:
        self.q_fine = q_fine
        # total flow across the field (c)
        self.tol = 1e-5
        self.c = np.sum(self.q[self.q>self.tol])                 
            
        # injectors
        self.n_inj = q_fine[q_fine>self.tol].size                          
        self.i_x, self.i_y = np.where(q_fine>self.tol)[0], np.where(q_fine>self.tol)[1]

        # producers
        self.n_prod = q_fine[q_fine<-self.tol].size
        self.p_x, self.p_y =  np.where(q_fine<-self.tol)[0], np.where(q_fine<-self.tol)[1]

        # policy_action and observation spaces
        self.observation_space = spaces.Box(low=np.array([-1]*(2*self.n_prod+self.n_inj), dtype=np.float64), 
                                            high=np.array([1]*(2*self.n_prod+self.n_inj), dtype=np.float64), 
                                            dtype=np.float64)
        
        self.action_space = spaces.Box(low=np.array([0.001]*(self.n_prod+self.n_inj), dtype=np.float64), 
                                       high=np.array([1]*(self.n_prod+self.n_inj), dtype=np.float64), 
                                       dtype=np.float64)

    def define_accmap(self):
        L = len(self.level_dict)
        fine_grid = Grid(nx=self.level_dict[L][0],
                         ny=self.level_dict[L][1],
                         lx=self.grid.lx,
                         ly=self.grid.ly)
        self.accmap = get_accmap(fine_grid, self.grid)

    def set_k(self, k):
        assert k.ndim==3, 'Invalid value k. n permeabilities should be provided as a numpy array with shape (n,grid.nx, grid.ny)'
        self.k_list = k

class RessimEnvParamGenerator():
    def __init__(self,
                 ressim_params: RessimParams) -> None:

        """
        a parameter generator for the MultiLevelRessimEnv environment

        :param ressim_params: reservoir simulation parameters
        :param level_dict: level dictionary

        """

        # check the level_dict input
        for i,l in enumerate(ressim_params.level_dict.keys()):
            assert i+1==l, 'level_dict keys should start from one to the lenth of the dictionary'
            if i>0:
                assert sum(ressim_params.level_dict[l]) > sum(ressim_params.level_dict[l-1]), 'level_dict values should reflect grid dimensions in ascending order such that the last value is one'

        self.ressim_params = ressim_params

    def get_level_env_params(self, level: int):
        assert level in self.ressim_params.level_dict.keys(), 'invalid level value, should be among the level_dict keys'
        coarse_grid = Grid(nx=self.ressim_params.level_dict[level][0], 
                           ny=self.ressim_params.level_dict[level][1],
                           lx=self.ressim_params.grid.lx,
                           ly=self.ressim_params.grid.ly)
        accmap = get_accmap(self.ressim_params.grid, coarse_grid)
        coarse_phi = fine_to_coarse_mapping(self.ressim_params.phi, accmap, func=mean)
        coarse_q = fine_to_coarse_mapping(self.ressim_params.q, accmap, func=sum)
        coarse_s = fine_to_coarse_mapping(self.ressim_params.s, accmap, func=mean)
        coarse_k = []
        for k in self.ressim_params.k_list:
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
                               coarse_s,
                               self.ressim_params.level_dict))

        ressim_params_coarse = RessimParams(*coarse_params)
        ressim_params_coarse.define_obs_act_spaces(self.ressim_params.q)

        return ressim_params_coarse


class MultiLevelRessimEnv(gym.Env):
    def __init__(self,
                 ressim_params: RessimParams,
                 level: int) -> None:
        
        assert level in ressim_params.level_dict.keys(), 'invalid level value, should be among the level_dict keys'

        self.ressim_params = ressim_params
        self.grid = self.ressim_params.grid
        self.level = level
        
        # RL parameters ( accordind to instructions on: https://github.com/openai/gym/blob/master/gym/core.py )
        self.metadata = {'render.modes': []} 
        self.reward_range = (0.0, 1.0)
        self.spec = None

        # define observation and policy_action spaces
        self.observation_space = self.ressim_params.observation_space
        self.action_space = self.ressim_params.action_space

        # for reproducibility
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def phi_a(self, policy_action):
        
        # convert input array into producer/injector 
        inj_flow = policy_action[:self.ressim_params.n_inj] / np.sum(policy_action[:self.ressim_params.n_inj])
        inj_flow = self.ressim_params.c * inj_flow
        prod_flow = policy_action[self.ressim_params.n_inj:] / np.sum(policy_action[self.ressim_params.n_inj:])
        prod_flow = -self.ressim_params.c * prod_flow
        
        # add producer/injector flow values
        q = np.zeros(self.ressim_params.q_fine.shape)
        q[self.ressim_params.i_x, self.ressim_params.i_y] = inj_flow
        q[self.ressim_params.p_x, self.ressim_params.p_y] = prod_flow

        # adjust unbalanced source term in arbitary location in the field due to precision error 
        if np.abs(np.sum(q)) < self.ressim_params.tol:
            q[3,3] = q[3,3] - np.sum(q) 

        return q

    def phi_a_inverse(self, q):
        inj_flow = q[self.ressim_params.i_x, self.ressim_params.i_y]
        inj_flow = inj_flow/np.sum(inj_flow)
        prod_flow = q[self.ressim_params.p_x, self.ressim_params.p_y]
        prod_flow = prod_flow/np.sum(prod_flow)
        return np.hstack((np.abs(inj_flow), np.abs(prod_flow)))

    def Phi_a(self, q) -> None:
        self.env_action = fine_to_coarse_mapping(q, self.ressim_params.accmap, func=sum)
        
    def Phi_s(self) -> None:
        s_fine = coarse_to_fine_mapping(self.state['s'], self.ressim_params.accmap)
        p_fine = coarse_to_fine_mapping(self.state['p'], self.ressim_params.accmap)
        return s_fine, p_fine

    def phi_s(self, s_fine, p_fine):
        obs_sat = s_fine[self.ressim_params.p_x, self.ressim_params.p_y]

        # scale pressure into the range [-1,1]
        fine_p_scaled = np.interp(p_fine, (p_fine.min(), p_fine.max()), (-1,1))
        obs_pr_p = fine_p_scaled[self.ressim_params.p_x, self.ressim_params.p_y]
        obs_pr_i = fine_p_scaled[self.ressim_params.i_x, self.ressim_params.i_y]

        self.obs = np.hstack((obs_sat, obs_pr_p, obs_pr_i))

    def simulation_step(self):
        # solve pressure
        self.solverP = PressureEquation(self.ressim_params.grid, q=self.env_action, k=self.k_load, lamb_fn=self.ressim_params.lamb_fn)
        self.solverS = SaturationEquation(self.ressim_params.grid, q=self.env_action, phi=self.ressim_params.phi, s=self.state['s'], f_fn=self.ressim_params.f_fn, df_fn=self.ressim_params.df_fn)

        # solve pressure equation
        oil_pr = 0.0
        self.solverP.s = self.solverS.s
        self.solverP.step()
        self.solverS.v = self.solverP.v
        for _ in range(self.ressim_params.nstep):
            # solve saturation equation
            self.solverS.step(self.ressim_params.dt)
            oil_pr = oil_pr + -np.sum( self.env_action[self.env_action<0] * ( 1- self.ressim_params.f_fn(self.solverS.s[self.env_action<0]) ) )*self.ressim_params.dt

        # state        
        self.state['s'] = self.solverS.s
        self.state['p'] = self.solverP.p

        #reward
        reward = oil_pr / self.ressim_params.ooip # recovery rate

        # done
        self.episode_step += 1
        if self.episode_step >= self.ressim_params.terminal_step:
            done=True
        else:
            done=False

        return self.state, reward, done, {}

    def step(self, policy_action):
        q_fine = self.phi_a(policy_action)
        self.Phi_a(q_fine)
        _, reward, done, info = self.simulation_step()
        s_fine, p_fine = self.Phi_s()
        self.phi_s(s_fine, p_fine)
        return self.obs, reward, done, info

    def reset(self):

        self.env_action = self.ressim_params.q

        # initialize dynamic parameters
        s = self.ressim_params.s
        p = np.zeros(self.ressim_params.grid.shape)
        k_index = self.np_random.choice(self.ressim_params.k_list.shape[0])
        e = 0
        self.set_dynamic_parameters(s,p,k_index,e)

        s_fine, p_fine = self.Phi_s()
        self.phi_s(s_fine, p_fine)

        return self.obs

    def update_current_obs(self):
        s_fine, p_fine = self.Phi_s()
        self.phi_s(s_fine, p_fine)

    def set_dynamic_parameters(self, s, p, k_index, e):
        # dynamic parameters
        self.state = {'s':s, 'p':p}
        self.k_index = k_index
        self.k_load = self.ressim_params.k_list[self.k_index]
        self.episode_step = e

    def map_from_(self, grid, level, state, k_index, episode_step):

        grid_from, level_from = grid, level
        grid_to, level_to = self.ressim_params.grid, self.level
        
        if level_from > level_to:
            # fine to coarse mapping
            accmap = get_accmap(grid_from, grid_to)
            s = fine_to_coarse_mapping(state['s'], accmap, func=mean)
            p = fine_to_coarse_mapping(state['p'], accmap, func=mean)
            self.set_dynamic_parameters(s,p,k_index,episode_step)
            self.update_current_obs()
        else:
            # coarse to fine mapping
            accmap = get_accmap(grid_to, grid_from)
            s = coarse_to_fine_mapping(state['s'], accmap)
            p = coarse_to_fine_mapping(state['p'], accmap)
            self.set_dynamic_parameters(s,p,k_index,episode_step)
            self.update_current_obs()


    def map_from(self, env):

        grid = env.ressim_params.grid
        level = env.level
        state = env.state
        k_index = env.k_index
        episode_step = env.episode_step

        self.map_from_(grid, level, state, k_index, episode_step)

    def set_k(self, k):
        self.ressim_params.set_k(k)
