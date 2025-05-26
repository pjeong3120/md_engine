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
    # Initialize Lattice
    temp = 2.5
    unit_cell = np.array([10, 10])
    initialization_lattice_cell = np.array([2, 2])
    r0, v0 = initialize_particles(temp, unit_cell, initialization_lattice_cell)

    masses = np.ones((r0.shape[0]))
    r_max = 5.0
    sigma = 1.0
    epsilon = 1.0
    lj_potential = LennardJones(sigma, epsilon, unit_cell, r_max)

    total_time = 10

    # Generate data
    for i, dt in enumerate([0.01, 0.001, 0.0001, 0.00001]):
        r = r0.copy()
        v = v0.copy()
        num_steps = int(total_time / dt)
        save_every = 10 ** i

        engine = VerletEngine(r, v, masses, lj_potential, dt, unit_cell)
        data = engine.run(num_steps, save_every, job_name = f'totaltime=10_dt=e-{i+2}')
    
    # Generate images + gifs
    for i, dt in enumerate([0.01, 0.001, 0.0001, 0.00001]):
        frame_paths = visualize_trajectory(f'./runs/totaltime=10_dt=e-{i+2}', visualize_up_to = 101)
        make_gif(frame_paths, out_path = f'./runs/totaltime=10_dt=e-{i+2}/data.gif')
