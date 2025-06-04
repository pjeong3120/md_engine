import numpy as np
import pickle
import os
from src.utils import get_distance_matrices_pbc
from potentials import Potential


def compute_pressure(masses : np.ndarray, 
                     r : np.ndarray, 
                     v : np.ndarray, 
                     potential : Potential, 
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
