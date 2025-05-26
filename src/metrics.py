import numpy as np
import pickle
import os


def compute_traj_energy(data):
    pe = data['per_atom_pe'] # (T, N)
    masses = data['masses'] # (N)
    v = data['v'] # (T, N, 2)
    ke = masses / 2 * (v**2).sum(axis = -1)

    system_energy = (pe + ke).sum(axis = -1) # (T,)
    return system_energy


if __name__ == '__main__':
    pass