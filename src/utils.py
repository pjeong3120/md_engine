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
    - displacement_tensor (np.ndarray): Numpy tensor with shape (N, N, d). The entry at index [i, j, :] is equal to r[i] - r[j]
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


def compute_pressure(masses : np.ndarray, 
                     r : np.ndarray, 
                     v : np.ndarray, 
                     potential : 'Potential', 
                     unit_cell : np.ndarray):
    """
    Virial expression for pressure. Found on page 351 of Limmer textbook
    Note that the virial expression is only valid if the total forces sum to zero.
    For a Lennard Jones potential, this holds because Fij = Fji
    """
    d = unit_cell.shape[0]
    V = unit_cell.prod() 

    # Computing KE terms
    # m: (N,)
    # v: (N, d)
    ke_terms = (masses[:, np.newaxis] * v ** 2).sum()
    
    # Computing Virial Correction
    # per_atom_forces: (N, d)
    # r:  (N, d)
    displacement_tensor, distance_matrix = get_distance_matrices_pbc(r, unit_cell)
    system_pe, per_atom_pe, per_atom_force, atom_atom_force = potential.get_energy_force(r)
    # (N, N, d) (N, N, d)
    virial_correction = (atom_atom_force * displacement_tensor).sum() / 2
    return (ke_terms + virial_correction) / (d * V)

def compute_diffusion_coeff_r(time : np.ndarray, r : np.ndarray, include_entire_traj = False):
    # Note to self - make sure to deal with pbc crossings.
    # (T,)
    # (T, N, d)
    d = r.shape[-1]
    displacement = ((r[:,:,:] - r[0, :, :])**2).sum(axis = -1) # (T, N)
    diffusion_coeffs = (displacement / time[:, np.newaxis]) / d

    if include_entire_traj:
        return diffusion_coeffs.mean()
    else:
        return diffusion_coeffs[-1, :].mean()

def compute_diffusion_coeff_v(time : np.ndarray, v : np.ndarray):
    T, N, d = v.shape
    
    autocorrelation = (v[:, :, :] * v[0, :, :]).sum(axis = -1).sum(axis = -1) # First mean takes the autocorrelation. Second mean is over all particles
    integral_autocorrelation = (autocorrelation * time[:]).sum() # Sum over all time

    return integral_autocorrelation
