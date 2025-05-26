import numpy as np
import pickle
import os
from functools import partial
from abc import ABC, abstractmethod
from tqdm import tqdm
from potentials import Potential, LennardJones
from utils import check_pbc, initialize_particles

class Engine(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def step(self):
        pass




class VerletEngine(Engine):
    def __init__(self,
                 r0 : np.ndarray,
                 v0 : np.ndarray,
                 masses : np.ndarray,
                 potential : Potential,
                 dt : float,
                 unit_cell : np.ndarray):
        
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

        potential_energy, forces =  self.potential.get_energy_forces(self.r)
        self.v = self.v + forces * self.dt / 2 / self.masses[:, np.newaxis]
        self.r = self.r + self.v * self.dt 
        self.r = check_pbc(self.r, self.unit_cell)
        potential_energy, forces = self.potential.get_energy_forces(self.r)
        self.v = self.v + forces * self.dt / 2 / self.masses[:, np.newaxis]

        return self.r, self.v, potential_energy, forces
    
    def run(self, num_steps, save_every, out_dir = './runs', job_name = 'example_job'):
        num_saves = num_steps // save_every + 1

        self.data = {'time' : np.zeros((num_saves, )), 
                     'unit_cell' : self.unit_cell,
                     'masses' : self.masses,
                     'r' : np.zeros((num_saves, self.r.shape[0], self.r.shape[1])), # (T, N, 2)
                     'v' : np.zeros((num_saves, self.r.shape[0], self.r.shape[1])), # (T, N, 2)
                     'per_atom_pe' : np.zeros((num_saves, self.r.shape[0])), # (T, N)
                     'per_atom_force' : np.zeros((num_saves, self.r.shape[0], self.r.shape[1])) # (T, N, 2)
                     }

        potential_energy, forces =  self.potential.get_energy_forces(self.r)
        self.data['time'][0] = 0
        self.data['r'][0, :, :] = self.r
        self.data['v'][0, :, :] = self.v
        self.data['per_atom_pe'][0, :] = potential_energy
        self.data['per_atom_force'][0, :, :] = forces
        
        for i in tqdm(range(1, num_steps + 1)):
            r, v, potential_energy, forces = self.step()
            
            if i % save_every == 0:
                idx = i // save_every
                self.data['time'][idx] = i * self.dt
                self.data['r'][idx, :, :] = self.r
                self.data['v'][idx, :, :] = self.v
                self.data['per_atom_pe'][idx, :] = potential_energy
                self.data['per_atom_force'][idx, :, :] = forces

        out_dir = os.path.join(out_dir, job_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(os.path.join(out_dir, 'data.pkl'), 'wb') as file:
            pickle.dump(self.data, file)
        
        return self.data



if __name__ == '__main__':
    # Initialize Lattice
    temp = 2.5
    unit_cell = np.array([10, 10])
    initialization_lattice_cell = np.array([2, 2])
    r0, v0 = initialize_particles(temp, unit_cell, initialization_lattice_cell)
    masses = np.ones((r0.shape[0]))
    r_max = 5.0
    sigma = 1.0
    epsilon = 1.0
    lj_potential = LennardJones(sigma, epsilon, unit_cell, r_max)

    total_time = 10


    for i, dt in enumerate([0.01, 0.001, 0.0001, 0.00001]):
        r = r0.copy()
        v = v0.copy()
        num_steps = int(total_time / dt)
        save_every = 10 ** i

        engine = VerletEngine(r, v, masses, lj_potential, dt, unit_cell)
        data = engine.run(num_steps, save_every, job_name = f'dt_e-{i+1}')
