import numpy as np


def check_pbc(r : np.ndarray, unit_cell : np.ndarray):
    """
    Applies periodic boundary conditions to particle positions (r) according to unit_cell
    """
    return r - np.floor(r / unit_cell) * unit_cell


def get_distance_matrices_pbc(r : np.ndarray, unit_cell : np.ndarray):
    """
    Takes in a radius matrix with shape (N, 2) and returns distance/displacement vectors
    unit_cell is used to deal with pbc

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
    # Ensure average velocity has mean=0 (ie no net flow) and std=temp (ie has specified temperature)
    v = (v - v.mean(axis = 1, keepdims = True)) * temp / v.std()
    return r, v

