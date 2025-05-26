from abc import ABC, abstractmethod
import numpy as np
from src.utils import check_pbc, get_distance_matrices_pbc


class Potential(ABC):
    def __init__(self):
        """
        Potential parameterization
        """
        super().__init__()

    @abstractmethod
    def get_energy_forces(self, r):
        """
        Forward call of the potential
        """
        pass


class LennardJones(Potential):
    def __init__(self, 
                 sigma : float, 
                 epsilon : float,
                 unit_cell : np.ndarray,
                 r_max : float):
        super().__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.unit_cell = unit_cell
        self.r_max = r_max
    
    def get_energy_forces(self, r):
        """
        Computes the per atom potential energy and force.
        Periodic boundary conditions are applied when computing energies and forces

        Input: 
        - r (np.ndarray): Numpy matrix with shape (N, d) with the positions of each particle
        
        Returns:
        - potential_energy (np.ndarray): Numpy vector with shape (N,) with per-atom potential energy
        - forces (np.ndarray): Numpy matrix with shape (N, d) with per-atom forces
        """

        # Step 1 - Compute distances and directions for closest neighbors according to pbc
        displacement_tensor, distance_matrix = get_distance_matrices_pbc(r, self.unit_cell)

        # Step 2 - Create a mask to ignore self-interaction and r > r_max
        # Note that mask is (N, N)
        mask = np.logical_and(distance_matrix < self.r_max, distance_matrix > 0)

        # Step 3 - potential energy
        potential_energy = np.zeros_like(distance_matrix) # Fill with zeros - ignored terms should contribute zero 
        potential_energy[mask] = 4 * self.epsilon * ((self.sigma / distance_matrix[mask]) ** 12 - (self.sigma / distance_matrix[mask]) ** 6)
        potential_energy = potential_energy.sum(axis = 1)

        # Step 4 - forces
        forces = np.zeros_like(displacement_tensor)
        forces[mask] = 24 * self.epsilon * ((
                2 * (self.sigma / distance_matrix[mask]) ** 12 - 
                (self.sigma / distance_matrix[mask]) ** 6
                ) / (distance_matrix[mask] ** 2)
            )[:, np.newaxis] * displacement_tensor[mask]
        forces = forces.sum(axis = 1) # Sum along all atoms j 

        return potential_energy, forces





