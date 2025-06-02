from src.utils import initialize_n_particles_target_temp_2d
from src.potentials import LennardJones
from src.engine import VerletEngine
from src.visualize import visualize_trajectory, make_gif
from src.utils import k_B
import numpy as np

import pickle, os, numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    np.random.seed(42)
    # Initialize Lattice
    target_temp = 5 
    N = 64
    unit_cell = np.array([10, 10])
    masses = np.ones((N,))

    r0, v0 = initialize_n_particles_target_temp_2d(N, masses, target_temp, unit_cell)

    # Parameterize potential
    r_max = 5.0
    sigma = 1.0
    epsilon = 1.0
    lj_potential = LennardJones(sigma, epsilon, unit_cell, r_max)

    # screen through dt - energy.std() = dt^2
    total_time = 10
    for i, dt in enumerate([0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010]):
        r = r0.copy()
        v = v0.copy()
        num_steps = int(total_time / dt)
        save_every = max(1, num_steps // 1000) # Save 1000 frames total

        engine = VerletEngine(r, v, masses, lj_potential, dt, unit_cell)
        if i == 0:
            job_name = f'N={int(r.shape[0])}_totaltime={int(total_time)}_dt=1e-4'
        else:
            job_name = f'N={int(r.shape[0])}_totaltime={int(total_time)}_dt={i*2}e-4'

        data = engine.run(num_steps, save_every, job_name = job_name)

        # Visualize Data 
        frame_paths = visualize_trajectory(job_path=f'./runs/{job_name}', visualize_up_to = 101) # Visualize only 101 frames for runtime aka only the first 1 s
        make_gif(frame_paths, job_path = f'./runs/{job_name}')
