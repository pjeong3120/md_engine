import numpy as np
import pickle
import os
from functools import partial
from abc import ABC
from tqdm import tqdm




def check_pbc(r : np.ndarray, unit_cell : np.ndarray):
    """
    Applies periodic boundary conditions to particle positions (r) according to unit_cell
    """
    return r - np.floor(r / unit_cell) * unit_cell



def get_distance_matrices_pbc(r : np.ndarray, unit_cell : np.ndarray):
    """
    Takes in a radius matrix with shape (N, 2) and returns distance/displacement vectors
    unit_cell is used to deal with pbc
    TODO - write tests to make sure this works!

    Input: 
    - r (np.ndarray): Numpy matrix with shape (N, d) containing the positions of each particle
    - unit_cell(np.ndarray): Numpy vector with shape (d,) containing the side lengths of the unit cell

    Returns:
    - displacement_vectors (np.ndarray): Numpy tensor with shape (N, N, d). The entry at index [i, j, :] is equal to r[i] - r[j]
    - distance_matrix (np.ndarray): Numpy tensor with shape (N, N, d). The entry at index[i, j] is the distane from r[i] to r[j]
    """

    r = check_pbc(r, unit_cell)

    displacement_tensor = r[:, np.newaxis, :] - r[np.newaxis, :, :] # (N, N, 2), displacement_vectors[i, j, :] = r[i] - r[j]
    displacement_tensor = displacement_tensor - unit_cell * np.round(displacement_tensor / unit_cell)

    distance_matrix = np.sqrt(np.sum(np.square(displacement_tensor), axis = -1)) # (N, N)

    return displacement_tensor, distance_matrix



def lennard_jones_potential(r : np.ndarray, 
                            sigma : float, 
                            epsilon : float,
                            unit_cell : np.ndarray,
                            r_max : float):
    """
    Computes the per atom potential energy and force.
    Periodic boundary conditions are applied when computing energies and forces

    Input: 
    - r (np.ndarray): Numpy matrix with shape (N, d) with the positions of each particle
    - sigma (float): Potential well radius
    - epsilon (float): Depth of the potential well
    - unit_cell (np.ndarray): Numpy vector with shape (d,) with rectangular unit cell lengths
    - r_max (float): Radius cutoff for force computations

    Returns:
    - potential_energy (np.ndarray): Numpy vector with shape (N,) with per-atom potential energy
    - forces (np.ndarray): Numpy matrix with shape (N, d) with per-atom forces
    """

    # Step 1 - Compute distances and directions for closest neighbors according to pbc
    displacement_tensor, distance_matrix = get_distance_matrices_pbc(r, unit_cell)

    # Step 2 - Create a mask to ignore self-interaction and r > r_max
    # Note that mask is (N, N)
    mask = np.logical_and(distance_matrix < r_max, distance_matrix > 0)

    # Step 3 - potential energy
    potential_energy = np.zeros_like(distance_matrix) # Fill with zeros - ignored terms should contribute zero 
    potential_energy[mask] = 4 * epsilon * ((sigma / distance_matrix[mask]) ** 12 - (sigma / distance_matrix[mask]) ** 6)
    potential_energy = potential_energy.sum(axis = 1)

    # Step 4 - forces
    forces = np.zeros_like(displacement_tensor)
    forces[mask] = 24 * epsilon * ((
            2 * (sigma / distance_matrix[mask]) ** 12 - 
            (sigma / distance_matrix[mask]) ** 6
            ) / (distance_matrix[mask] ** 2)
        )[:, np.newaxis] * displacement_tensor[mask]
    forces = forces.sum(axis = 1) # Sum along all atoms j 

    return potential_energy, forces




def verlet_step(r : np.ndarray, 
                v : np.ndarray, 
                masses : np.ndarray, 
                potential : callable, 
                dt : float, 
                unit_cell : np.ndarray):
    """
    Computes the particle positions (r) and velocities (v) according to a Verlet integration step.
    Usage - verlet step will call potential(r) so the potential function needs to be a function of 
    particle positions only. Use partial to parameterize potentials as necessary. 

    Inputs:
    - r (np.ndarray): Numpy matrix with shape (N, d) with the positions of each particle
    - v (np.ndarray): Numpy matrix with shape (N, d) with the velocities of each particle
    - masses (np.ndarray): Numpy vector with shape (N,) with the masses of each particle
    - potential (function): A potential function that returns per-atom potential energy and forces as a function of particle positions (r)
    - dt (float): Discrete time step
    - unit_cell (np.ndarray): Numpy vector with shape (d,) with rectangular unit cell lengths
    """


    potential_energy, forces =  potential(r)
    v = v + forces * dt / 2 / masses[:, np.newaxis]
    r = r + v * dt 
    r = check_pbc(r, unit_cell)
    potential_energy, forces = potential(r)
    v = v + forces * dt / 2 / masses[:, np.newaxis]

    return r, v, potential_energy, forces




def initialize_particles(temp : float,
                         unit_cell : np.ndarray,
                         initialization_lattice_cell : np.ndarray
                         ):
    """
    Initializes a regular lattice of particles across a unit_cell. 
    TODO - do some math to figure out conversion between temp and <v^2>


    Inputs:
    - temp (float): standard deviation of initial velocity
    - unit_cell (np.ndarray): Numpy vector with shape (d,) with rectangular unit cell lengths
    - initialization_lattice_cell (np.ndarray): Distance between particles at t0

    Outputs:
    - r (np.ndarray): Particle Positions
    - v (np.ndarray): Particle velocities
    """


    sigma = 0.001
    dx = initialization_lattice_cell[0]
    dy = initialization_lattice_cell[1]
    Nx = int(unit_cell[0] / dx)
    Ny = int(unit_cell[1] / dy)
    r = np.zeros((Nx, Ny, 2)) # (Nx, Ny, 2)

    for i in range(Nx):
        for j in range(Ny):
            r[i, j, :] = (i * dx + sigma, j * dy + sigma)
    

    r = r.reshape(Nx * Ny, 2) # (N, 2)
    r = check_pbc(r, unit_cell)

    v = np.random.normal(loc = 0, scale = temp, size = r.shape)
    # Ensure average velocity has mean=0 (ie no net flow) and std=temp (ie has specifiedtemperature)
    v = (v - v.mean(axis = 1, keepdims = True)) * temp / v.std()
    return r, v
    



def run_dynamics(r, v, masses, potential, dt, num_steps, unit_cell):

    data = {'unit_cell' : unit_cell,
            'masses' : masses,
            'r' : np.zeros((num_steps + 1, r.shape[0], r.shape[1])), # (T, N, 2)
            'v' : np.zeros((num_steps + 1, r.shape[0], r.shape[1])), # (T, N, 2)
            'per_atom_pe' : np.zeros((num_steps + 1, r.shape[0])), # (T, N)
            'per_atom_force' : np.zeros((num_steps + 1, r.shape[0], r.shape[1])) # (T, N, 2)
            }

    potential_energy, forces =  potential(r)
    data['r'][0, :, :] = r
    data['v'][0, :, :] = v
    data['per_atom_pe'][0, :] = potential_energy
    data['per_atom_force'][0, :, :] = forces
    
    for i in tqdm(range(1, num_steps + 1)):
        r, v, potential_energy, forces = verlet_step(r, v, masses, potential, dt, unit_cell)
        data['r'][i, :, :] = r
        data['v'][i, :, :] = v
        data['per_atom_pe'][i, :] = potential_energy
        data['per_atom_force'][i, :, :] = forces

    return data


def serialize(data, out_dir = './runs', job_name = 'example_job'):
    out_dir = os.path.join(out_dir, job_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, 'data.pkl'), 'wb') as file:
        pickle.dump(data, file)
    



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
    lj_potential = partial(lennard_jones_potential, sigma = sigma, epsilon = epsilon, unit_cell = unit_cell, r_max = r_max)


    for i, dt in enumerate([0.01, 0.001, 0.0001]):
        r = r0.copy()
        v = v0.copy()
        num_steps = int(10 / dt)

        data = run_dynamics(r, v, masses, lj_potential, dt, num_steps, unit_cell)
        serialize(data, job_name = f'dt_e-{i+1}')