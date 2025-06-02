import numpy as np

k_B = 1


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



def compute_temperature(masses : np.ndarray, v : np.ndarray):
    """
    Computes temperature from particle masses and velocities.

    Parameters:
    - masses: np.ndarray of shape (N,)
    - velocities: np.ndarray of shape (N, d)
    - k_B: Boltzmann constant (default in J/K)

    Returns:
    - temperature in Kelvin
    """
    kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * v**2)
    N, d = v.shape
    temperature = (2 * kinetic_energy) / (N * d * k_B)
    return temperature



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

