import numpy as np
import pickle
import os


def compute_traj_energy(data):
    pe = data['per_atom_pe'].sum(axis = -1) # (T, N) -> (T)
    masses = data['masses'] # (N)
    v = data['v'] # (T, N, 2)
    ke = (masses / 2 * (v**2).sum(axis = -1)).sum(axis = -1) #(T, N, 2) -> (T)

    system_energy = (pe + ke)
    return system_energy, pe, ke


if __name__ == '__main__':
    pass