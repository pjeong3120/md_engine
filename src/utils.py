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
def bin_values(values : np.ndarray, 
        num_bins : int, 
        return_edges = False):
    """
    Helper function for converting an array of values into a distribution.

    Parameters:
    - values (np.ndarray): Numpy array with shape (N,)
    - num_bins (int): Number of bins
    - return_edges (bool, default = False): Whether to return bin edges or bin centers.
        For most purposes, including plotting, bin centers is more useful. 

    Returns:
    - bin_centers or bin_edges (np.ndarray): Self explanatory. Shape is (num_bins,) for 
    bin_centers and (num_bins + 1,) for bin_edges
    - bin_heights (np.ndarray): The total number of elements in each bin. bin_heights is
        not normalized.
    """
    
    if values.ndim != 1:
        raise ValueError(f"Unsupported input values ndim: {values.ndim}. Only supports vectors")
    
    # Step 1: Compute the bin length, bin edges, and the bin that each value belongs to.
    # TODO - check if behavior is the same for (N,)
    bin_length = (values.max() - values.min()) / num_bins
    bin_edges = np.linspace(values.min(), values.max(), num_bins + 1)

    # Step 2: Compute which bin each value belongs to
    bin_indices = np.floor((values - values.min()) / bin_length).astype(int)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1) # Guarantees that we don't run into indexing issues for values.max().
    
    # Step 3: Compute bin heights
    bin_heights = np.zeros(num_bins)
    np.add.at(bin_heights, bin_indices, 1)  # Safer than +=

    # Step 4: Return
    if return_edges:
        return bin_edges, bin_heights
    
    bin_centers = bin_edges[:-1] + bin_length / 2

    return bin_centers, bin_heights

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
      an array with shape (T,) with temperatures across all time steps. Otherwise, it
      returns a single float. 
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


def normalized_radial_density_function(r : np.ndarray, 
                                       unit_cell : np.ndarray, 
                                       num_bins : int):
    """
    Computes g(r), which is the normalized density function. Returns the centers and heights
    of a histogram containing the binned g distribution. 

    Parameters:
    - r (np.ndarray): Array of shape (N, d) or (T, N, d), containing the positions of N particles. 
    - unit_cell (np.ndarray): side lengths of rectangular unit cell

    Returns:
    bin_centers (np.ndarray): 
    g (np.ndarray): 
    """
    # Step 1: Compute P(R)
    if r.ndim == 2:
        N, d = r.shape
        mask = np.triu_indices(N, k = 1)
        _, radii = get_distance_matrices_pbc(r, unit_cell)
        radii = radii[mask]

    elif r.ndim == 3:
        T, N, d = r.shape
        mask = np.triu_indices(N, k = 1)
        _, radii = get_distance_matrices_pbc(r, unit_cell)
        radii = radii[:, mask[0], mask[1]]
        radii = radii.reshape(-1)  # Flatten across time
    else:
        raise ValueError(f"Unsupported input r shape {r.shape}")
    
    mask = radii < (unit_cell.min() / 2)  # It only makes sense to compute it up to r=L/2 since greater 
                                          # than that the particles interact with themselves through the periodic image.
    radii = radii[mask]

    bin_centers, bin_heights = bin(radii, num_bins = num_bins)
    bin_length = bin_centers[1] - bin_centers[0]
    P_R = bin_heights / bin_heights.sum()  # Normalize to probability P(R)

    # Step 2: particle density 
    particle_density = N / unit_cell.prod()

    # Step 3: Compute shell area
    if d == 2:
        shell_area = 2 * np.pi * bin_centers * bin_length
    elif d == 3:
        shell_area = 4 * np.pi * bin_centers**2 * bin_length
    else:
        raise ValueError(f"Unsupported dimension d={d}")

    # Step 4: g = (N - 1) * P(R) / (shell_area * rho)
    g = P_R * (mask.sum() -1) / (shell_area * particle_density)

    return bin_centers, g


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


def velocity_autocorrelation_function(v : np.ndarray, lag : int):
    T = v.shape[0]
    autocorrelation = (v[lag: , :, :] * v[:T-lag]).sum(axis = -1)
    return autocorrelation.mean()    



def compute_diffusion_coeff_v(v : np.ndarray, time : np.ndarray):
    """
    Computes the diffusion coefficient from a 
    """
    T = v.shape[0]
    d = v.shape[2]
    autocorrelations = np.zeros((T - 1,))
    for i in range(T - 1):
        autocorrelations[i] = (velocity_autocorrelation_function(v, i))

    integral_autocorrelations = ((time[1:] - time[:-1]) * autocorrelations).sum()
    return integral_autocorrelations / d