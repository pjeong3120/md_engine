from src.potentials import LennardJones
from src.engine import MicrocanonicalVerletEngine, CanonicalVerletEngine, initialize_n_particles_target_temp_2d
from src.visualize import visualize_trajectory, make_gif
import numpy as np


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
    dt = 0.0010
    num_steps = int(total_time / dt)
    save_every = max(1, num_steps // 1000) # Save 1000 frames total


    for gamma in [0.001, 0.01, 0.1, 1.0, 10, 100]:
        
        r = r0.copy()
        v = v0.copy()
        
        engine = CanonicalVerletEngine(r, v, masses, lj_potential, dt, unit_cell, target_temp, gamma)

        job_name = f'gamma_screens/gamma={gamma}'

        data = engine.run(num_steps, save_every, job_name = job_name)
        print(data['temp'])
        # Visualize Data 
        frame_paths = visualize_trajectory(job_path=f'./runs/{job_name}', visualize_up_to = 101) # Visualize only 101 frames for runtime aka only the first 1 s
        make_gif(frame_paths, job_path = f'./runs/{job_name}')
