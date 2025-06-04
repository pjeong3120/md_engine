import numpy as np
import pickle
import os
from functools import partial
from abc import ABC, abstractmethod
from tqdm import tqdm
from src.potentials import Potential, LennardJones
from src.utils import check_pbc, k_B, compute_temperature


# Initialization schemes
# Initialize with a target temperature and num_particles

def initialize_n_particles_target_temp_2d(N : int,
                                          masses : np.ndarray,
                                          target_temp : float, 
                                          unit_cell : np.ndarray):
    """
    Initialize a regular lattice of particles across a unit cell.

    Inputs:
    - Temp (float): Temperature of the system
    - N (int): Number of simulated particles
    - unit_cell (np.ndarray): A numpy vector with size (2,) specifying the dimensions of a rectangular unit cell

    Returns:
    - r (np.ndarray): Numpy vector with shape (N, 2) specifying uniformly distributed particle positions across the unit cell
    - v (np.ndarray): NUmpy vector with shape (N, 2) specifying Boltzman distribution of particle velocities, parameterized by Temp
    """

    # If N is a perfect square, Nx and Ny will be the same.
    # Otherwise, Ny = Nx + 1 ie one more row
    Nx = np.floor(np.sqrt(N))
    Ny = np.ceil(np.sqrt(N))

    dx = unit_cell[0] / Nx
    dy = unit_cell[1] / Ny

    r = np.zeros((N, 2))

    for i in range(N):
        r[i, :] = ((i % Nx + 0.5) * dx, (i // Ny + 0.5) * dy)
    
    r = check_pbc(r, unit_cell)

    v = np.random.normal(loc = 0, scale = 1.0, size = r.shape)
    v = v - v.mean(axis=0, keepdims=True) # Remove any net velocity
    temp = compute_temperature(masses, v)
    v = v * np.sqrt(target_temp / temp) # 
    return r, v




class Engine(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def step(self):
        pass

    def run(self, 
            num_steps, 
            save_every, 
            out_dir = './runs', 
            job_name = 'example_job'):
        
        num_saves = num_steps // save_every + 1

        self.data = {'unit_cell' : self.unit_cell,
                     'masses' : self.masses,
                     'potential' : self.potential,
                     'time' : np.zeros((num_saves, )), 
                     'r' : np.zeros((num_saves, self.r.shape[0], self.r.shape[1])), # (T, N, 2)
                     'v' : np.zeros((num_saves, self.r.shape[0], self.r.shape[1])), # (T, N, 2)
                     'per_atom_pe' : np.zeros((num_saves, self.r.shape[0])), # (T, N)
                     'per_atom_ke' : np.zeros((num_saves, self.r.shape[0])), # (T, N)
                     'per_atom_force' : np.zeros((num_saves, self.r.shape[0], self.r.shape[1])), # (T, N, 2)
                     'system_pe' : np.zeros((num_saves,)),  # (T)
                     'system_ke' : np.zeros((num_saves,)),  # (T)
                     'system_energy' : np.zeros((num_saves)), # (T)
                     'temperature' : np.zeros((num_saves)) # (T)
                     }

        system_pe, per_atom_pe, per_atom_force, atom_atom_force =  self.potential.get_energy_force(self.r)
        self.system_pe = system_pe
        self.per_atom_pe = per_atom_pe
        self.per_atom_force = per_atom_force

        self.update_data(0, 0)
        print(f"Beginning job {job_name}")
        for t in tqdm(range(1, num_steps + 1)):
            self.step()
            
            if t % save_every == 0:
                idx = t // save_every
                self.update_data(t, idx)
        
        out_dir = os.path.join(out_dir, job_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(os.path.join(out_dir, 'data.pkl'), 'wb') as file:
            pickle.dump(self.data, file)
        
        return self.data
    
    
    def update_data(self, t, idx):
        self.data['time'][idx] = t * self.dt
        self.data['r'][idx, :, :] = self.r
        self.data['v'][idx, :, :] = self.v
        self.data['per_atom_pe'][idx, :] = self.per_atom_pe
        self.data['per_atom_ke'][idx, :] = self.masses / 2 * (self.v**2).sum(axis = -1)
        self.data['per_atom_force'][idx, :, :] = self.per_atom_force
        self.data['system_pe'][idx] = self.system_pe
        self.data['system_ke'][idx] = self.data['per_atom_ke'][idx].sum()
        self.data['system_energy'][idx] = self.data['system_pe'][idx] + self.data['system_ke'][idx]
        self.data['temperature'][idx] = compute_temperature(self.masses, self.v)

class MicrocanonicalVerletEngine(Engine):
    def __init__(self,
                 r0 : np.ndarray,
                 v0 : np.ndarray,
                 masses : np.ndarray,
                 potential : Potential,
                 dt : float,
                 unit_cell : np.ndarray):
        super().__init__()

        self.r = r0.copy()
        self.v = v0.copy()
        self.masses = masses.copy()
        self.potential = potential
        self.dt = dt
        self.unit_cell = unit_cell
        

    def step(self):
        """
        Updates the particle positions (r) and velocities (v) according to a Verlet integration step.
        """
        # Step 1: Velocity half step
        system_pe, per_atom_pe, per_atom_force, atom_atom_force =  self.potential.get_energy_force(self.r)
        self.v = self.v + per_atom_force * self.dt / 2 / self.masses[:, np.newaxis]
        
        # Step 2: Position step
        self.r = self.r + self.v * self.dt 
        self.r = check_pbc(self.r, self.unit_cell)
        
        # Step 3: Velocity half step
        system_pe, per_atom_pe, per_atom_force, atom_atom_force = self.potential.get_energy_force(self.r)
        self.v = self.v + per_atom_force * self.dt / 2 / self.masses[:, np.newaxis]

        # Update energies
        self.system_pe = system_pe
        self.per_atom_pe = per_atom_pe
        self.per_atom_force = per_atom_force



class CanonicalVerletEngine(Engine):
    def __init__(self,
                 r0 : np.ndarray,
                 v0 : np.ndarray,
                 masses : np.ndarray,
                 potential : Potential,
                 dt : float,
                 unit_cell : np.ndarray,
                 T : float,
                 gamma : float):
        super().__init__()

        self.r = r0.copy()
        self.v = v0.copy()
        self.masses = masses.copy()
        self.potential = potential
        self.dt = dt
        self.unit_cell = unit_cell
        self.T = T
        self.gamma = gamma

        self.a = np.exp(-gamma * dt / 2 / masses) # (N,)
        self.noise_std = np.sqrt(k_B * T * (1 - self.a **2) / masses)[:, np.newaxis] # (N, 1)
        # The new axis above allows us to broadcast later on when we sample. 
        # This works: normal(loc = 0, scale = (N, 1), size = (N, d))
        # This doesn't work: normal(loc = 0, scale = (N,), size = (N, d))

    
    def step(self):
        """
        Canonical Verlet Algorithm, as described on Page 356 of Limmer textbook. 
        The main difference between Canonical and Microcanonical is that at each 
        time step, a friction term reduces velocity and a temperature term 
        introduces a stochastic element so that the temperature remains constant.

        a = exp(-gamma dt / 2m)
        <R_{i, t}> = 0
        <R_{i, t}, R_{j, t'} = (k_B * T)/m_i * (1 - a**2) delta_{ij} delta(t - t')

        Some notes about R:
        - Mean is 0 ie no net flow
        - Variance is given as above. The delta functions simply guarantee that
        the stochastic term is independent between for each unique molecule and
        across time. 
        """

        # Step 1: Velocity half step
        R = np.random.normal(loc = 0, scale = self.noise_std, size = self.v.shape)
        system_pe, per_atom_pe, per_atom_force, atom_atom_force =  self.potential.get_energy_force(self.r)
        self.v = self.v * self.a + R + per_atom_force * self.dt / 2 / self.masses[:, np.newaxis]
        
        # Step 2: Position step
        self.r = self.r + self.v * self.dt 
        self.r = check_pbc(self.r, self.unit_cell)
        
        # Step 3: Velocity half step
        R = np.random.normal(loc = 0, scale = self.noise_std, size = self.v.shape)
        system_pe, per_atom_pe, per_atom_force, atom_atom_force = self.potential.get_energy_force(self.r)
        self.v = self.v * self.a + R + per_atom_force * self.dt / 2 / self.masses[:, np.newaxis]

        # Update energies
        self.system_pe = system_pe
        self.per_atom_pe = per_atom_pe
        self.per_atom_force = per_atom_force

        