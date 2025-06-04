from src.potentials import LennardJones
from src.engine import CanonicalVerletEngine, initialize_n_particles_target_temp_2d
from src.visualize import visualize_trajectory, make_gif
import numpy as np


if __name__ == '__main__':
    np.random.seed(42)

    # Initial lattice params
    target_temp = 5
    unit_cell = np.array([10, 10])

    # Parameterize potential
    r_max = 5.0
    sigma = 1.0
    epsilon = 10
    lj_potential = LennardJones(sigma, epsilon, unit_cell, r_max)
    
    # Other engine params
    dt = 0.001
    num_steps = int(10 / dt) # simulate for 10 seconds
    save_every = max(1, num_steps // 1000) # Save 1000 frames total
    gamma = 1.0


    for N in [16, 25, 36, 49, 64, 81, 85, 90, 95, 100, 105, 110, 115, 121]:
        masses = np.ones((N,))
        r, v = initialize_n_particles_target_temp_2d(N, masses, target_temp, unit_cell)

        engine = CanonicalVerletEngine(r, v, masses, lj_potential, dt, unit_cell, target_temp, gamma)
        
        job_name = f'density_screens/N={int(N)}'
        data = engine.run(num_steps, save_every, job_name = job_name)

        # Visualize Data 
        frame_paths = visualize_trajectory(job_path=f'./runs/{job_name}', visualize_up_to = 101) # Visualize only 101 frames for runtime aka only the first 1 s
        make_gif(frame_paths, job_path = f'./runs/{job_name}')
