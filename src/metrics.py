import numpy as np
import pickle
import os
from src.utils import get_distance_matrices_pbc



def pair_distribution(r, unit_cell, r_max):
    """
    Taken from Limmer textbook:
    For 3D: g(R) = (N-1) P(R) / (4 pi R^2 rho)
    For 2D: g(R) = (N-1) P(R) / (2 pi R rho)

    where N is the number of particles,
    P(R) is the radial probability distribution
    and rho is the bulk density (N / V)
    """

    N = r.shape[-2]
    rho = N / unit_cell.prod()

    # For computing P(R) - kind of funky!
    displacement_tensor, distance_matrix = get_distance_matrices_pbc(r, unit_cell)
    mask = np.logical_and(distance_matrix < r_max, np.triu_indices(N, k=1)) # Only take the upper right triangle for unique pairwise distances

    R = distance_matrix[mask]
    P_R = 1 / mask.sum() # P(R) 
    g_of_R = (N - 1) / (2 * np.pi * rho * R * mask.sum())

    return R, g_of_R

def binned_pair_pair_distribution(r, unit_cell, r_max, bins = 10):
    pass

if __name__ == '__main__':
    pass