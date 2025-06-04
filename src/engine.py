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
                     'system_energy' : np.zeros((num_saves)) # (T)
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





class VerletEngine(Engine):
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

        system_pe, per_atom_pe, per_atom_force, atom_atom_force =  self.potential.get_energy_force(self.r)
        self.v = self.v + per_atom_force * self.dt / 2 / self.masses[:, np.newaxis]
        self.r = self.r + self.v * self.dt 
        self.r = check_pbc(self.r, self.unit_cell)
        system_pe, per_atom_pe, per_atom_force, atom_atom_force = self.potential.get_energy_force(self.r)
        self.v = self.v + per_atom_force * self.dt / 2 / self.masses[:, np.newaxis]

        self.system_pe = system_pe
        self.per_atom_pe = per_atom_pe
        self.per_atom_force = per_atom_force



class LangevinVerletEngine(Engine):
    def __init__(self,
                 r0 : np.ndarray,
                 v0 : np.ndarray,
                 masses : np.ndarray,
                 potential : Potential,
                 dt : float,
                 unit_cell : np.ndarray,
                 T : float):
        super().__init__()

        self.r = r0.copy()
        self.v = v0.copy()
        self.masses = masses.copy()
        self.potential = potential
        self.dt = dt
        self.unit_cell = unit_cell
        self.T = T
    
    def step(self):
        system_pe, per_atom_pe, per_atom_force, atom_atom_force = self.potential.get_energy_force(self.r)

        a = per_atom_force / self.masses[:, np.newaxis]

        u1 = np.random.uniform(size = a.shape)
        u2 = np.random.uniform(size = a.shape)
        r1 = np.sqrt(k_B * self.T * (1 - a ** 2) / self.masses[:, np.newaxis] * (-2 * np.log(u1)) * np.cos(2 * np.pi * u2))
        r2 = np.sqrt(k_B * self.T * (1 - a ** 2) / self.masses[:, np.newaxis] * (-2 * np.log(u1)) * np.sin(2 * np.pi * u2))

        self.v = self.v * a + r1
        self.v = self.v + per_atom_force * self.dt * a / 2
        self.r = self.r + self.v * self.dt

        system_pe, per_atom_pe, per_atom_force, atom_atom_force =  self.potential.get_energy_force(self.r)
        a = per_atom_force / self.masses[:, np.newaxis]
        self.v = self.v + self.dt * a / 2
        self.v = self.v * a + r2

        self.system_pe = system_pe
        self.per_atom_pe = per_atom_pe
        self.per_atom_force = per_atom_force
        