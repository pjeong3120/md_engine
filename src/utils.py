"""
General use utility functions for molecular dynamics simulations. 
These functions support numpy tensors with shape (N, d) or (T, N, d), where 
N is the number of particles, d is the number of spatial dimensions, and T is the number of time steps.

"""


import numpy as np

k_B = 1


def check_pbc(r : np.ndarray, unit_cell : np.ndarray):
    """
    Applies periodic boundary conditions to particle positions (r) according to unit_cell

    Usage: r = check_pbc(r, unit_cell)
    """
    return r - np.floor(r / unit_cell) * unit_cell


def get_distance_matrices_pbc(r : np.ndarray, unit_cell : np.ndarray):
    """
    Computes displacement and distance matrices under PBC.

    Parameters:
    - r (np.ndarray): particle positions
    - unit_cell (np.ndarray): side lengths of rectangular unit cell

    Returns:
    - displacements (np.ndarray): shape (T, N, N, d) or (N, N, d)
    - distances (np.ndarray): shape (T, N, N) or (N, N)
    """

    r = check_pbc(r, unit_cell)

    if r.ndim == 2: # (N, d)
        displacements = r[:, np.newaxis, :] - r[np.newaxis, :, :]  # (N, N, d)
    elif r.ndim == 3: # (T, N, d)
        displacements = r[:, :, np.newaxis, :] - r[:, np.newaxis, :, :]  # (T, N, N, d)
    else:
        raise ValueError(f"Unsupported input r shape {r.shape}; must be (N, d) or (T, N, d)")
    
    displacements = displacements - unit_cell * np.round(displacements / unit_cell)
    distances = np.linalg.norm(displacements, axis=-1)  # (T, N, N) or # (N, N)
    return displacements, distances





"""
Thermodynamic and stat-mech properties
"""


def compute_temperature(masses : np.ndarray, v : np.ndarray):
    """
    Computes the instantaneous temperature of a system from particle velocities and masses.
    This assumes that all spatial degrees of freedom are active. 

    Parameters:
    - masses (np.ndarray): Array of shape (N,) containing the masses of N particles.
    - v (np.ndarray): Array of shape (N, d) or (T, N, d) containing particle velocities,
                      either for a single frame or across T time steps.
    
    Returns:
    - temperature (float): Temperature of system. If v has shape (T, N, d), this returns
      an array with shape (T,) with temperatures across all time steps. 
    """

    if v.ndim == 2: # (N, d)
        masses = masses[:, np.newaxis]
        N, d = v.shape
        kinetic_energy = 0.5 * (masses * v**2).sum() # Float
    elif v.ndim == 3: # (T, N, d)
        masses = masses[np.newaxis, :, np.newaxis]
        T, N, d = v.shape
        kinetic_energy = 0.5 * (masses * v**2).sum(axis = (1, 2)) # (T,) per frame KE
    else:
        raise ValueError(f"Unsupported input v shape {v.shape}; must be (N, d) or (T, N, d)")

    temperature = (2 * kinetic_energy) / (N * d * k_B) # N * d = dofs
    return temperature



def normalized_radial_density_function(r : np.ndarray, unit_cell : np.ndarray):
    """
    Computes the normalized radial density function g(r) under periodic boundary conditions.
    See Exercise 5.12 in the Limmer textbook.

    Parameters:
    - r (np.ndarray): Particle positions, shape (N, d) or (T, N, d), where d is 2 or 3.
    - unit_cell (np.ndarray): Side lengths of the rectangular unit cell, shape (d,)

    Returns:
    - distances (np.ndarray): Pairwise distances, shape (M,) or (T, M), with M = N(N-1)/2
    - g (np.ndarray): Normalized radial density values, same shape as distances
    """

    if r.ndim == 2:
        N, d = r.shape
        mask = np.triu_indices(N, k = 1)
        displacements, distances = get_distance_matrices_pbc(r, unit_cell)
        distances = distances[mask]

    elif r.ndim == 3:
        T, N, d = r.shape
        mask = np.triu_indices(N, k = 1)
        displacements, distances = get_distance_matrices_pbc(r, unit_cell)
        assert distances.ndim == 3 and distances.shape[0] == T and distances.shape[1] == N and distances.shape[2] == N, f'{distances.shape}'
        distances = distances[:, mask[0], mask[1]]
    else:
        raise ValueError(f"Unsupported input r shape {r.shape}; must be (N, d) or (T, N, d)")
    
    probabilities = 2 / (N * (N - 1)) # Each sample is equally likely => number of elements in the upper right triangle
    particle_density = N / unit_cell.prod()

    if d == 2:
        g = (N - 1) / (2 * np.pi * particle_density) * probabilities / distances 
    elif d == 3:
        g = (N - 1) / (4 * np.pi * particle_density) * probabilities / (distances ** 2)
    else:
        raise ValueError(f"Unsupported input r shape {r.shape}; last dim d must be 2 or 3")

    return distances, g


def binned_g(r : np.ndarray, unit_cell : np.ndarray, num_bins = 20, return_centers = True):
    radii, g_lst = normalized_radial_density_function(r, unit_cell)
    
    bin_length = radii.max() * 1.01 / num_bins
    bin_indices = np.floor(radii / bin_length).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_heights = np.zeros((num_bins,))
    bin_heights[bin_indices] += g_lst

    bin_edges = np.linspace(0, bin_length * num_bins, num_bins + 1)

    if return_centers:
        bin_centers = bin_edges[:-1] + bin_length / 2
        return bin_centers, bin_heights

    return bin_edges, bin_heights


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
    # Note that these will generally be wrong. There isn't a great way to compute 
    # 
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
