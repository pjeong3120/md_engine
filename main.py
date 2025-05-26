from src.utils import initialize_particles
from src.potentials import LennardJones
from src.engine import VerletEngine
from src.visualize import visualize_trajectory, make_gif
import numpy as np

"""
TODO - write main so that usage looks something like:
python main.py --mode {simulate/visualize} --job_name $JOB_NAME --params

"""


if __name__ == '__main__':
    np.random.seed(42)
    # Initialize Lattice
    temp = 2.5
    unit_cell = np.array([10.0, 10.0])
    initialization_lattice_cell = np.array([1.2, 1.2])

    r0, v0 = initialize_particles(temp, unit_cell, initialization_lattice_cell)

    # Parameterize potential
    masses = np.ones((r0.shape[0]))
    r_max = 5.0
    sigma = 1.0
    epsilon = 1.0
    lj_potential = LennardJones(sigma, epsilon, unit_cell, r_max)

    # screen through dt - energy.std() = dt^2
    total_time = 10
    for i, dt in enumerate([0.1, 0.01, 0.001, 0.0001, 0.00001]):
        r = r0.copy()
        v = v0.copy()
        num_steps = int(total_time / dt)
        save_every = max(1, num_steps // 1000) # Save 1000 frames total

        engine = VerletEngine(r, v, masses, lj_potential, dt, unit_cell)
        job_name = f'N={int(r.shape[0])}_totaltime={int(total_time)}_dt=1e{int(np.log10(dt))}'
        data = engine.run(num_steps, save_every, job_name = job_name)

        # Visualize Data 
        frame_paths = visualize_trajectory(job_path=f'./runs/{job_name}', visualize_up_to = 101) # Visualize only 101 frames for runtime
        make_gif(frame_paths, job_path = f'./runs/{job_name}')