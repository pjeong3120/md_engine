import matplotlib.pyplot as plt
import pickle
import os
import imageio.v2 as imageio
import glob
from tqdm import tqdm
import numpy as np

# TRAJECTORY UTILS
def visualize_single_frame(r, unit_cell, out_path = None):
    """
    Visualizes a single frame from the particle positions
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(r[:, 0], r[:, 1], s=50, edgecolor='k', facecolor='skyblue')
    plt.gca().set_aspect('equal')


    plt.xlim(0, unit_cell[0])
    plt.ylim(0, unit_cell[1])
    plt.gca().set_xticks([0, unit_cell[0]])
    plt.gca().set_yticks([0, unit_cell[1]])

    plt.grid(True, linestyle='--', alpha=0.5)
    
    if out_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(out_path))
    plt.close()



def visualize_trajectory(job_path, visualize_up_to = np.inf, visualize_every = 1):

    frames_dir = os.path.join(job_path, 'frames')
    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)

    with open(os.path.join(job_path, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    num_frames = min(data['r'].shape[0], visualize_up_to)
    unit_cell = data['unit_cell']
    frame_paths = []
    pad_width = len(str(num_frames - 1))
    for frame_idx in tqdm(range(0, num_frames, visualize_every)):
        out_path = os.path.join(frames_dir, f'{frame_idx:0{pad_width}d}.png')
        r = data['r'][frame_idx]
        visualize_single_frame(r, unit_cell, out_path = out_path)
        frame_paths.append(out_path)
    
    return frame_paths


def make_gif(frame_paths, job_path, fps=10):
    """
    Creates a gif from a sequence of PNG frames.

    Parameters:
    - frame_dir (str): Directory containing the PNG frames.
    - out_path (str): Output path for the gif file.
    - fps (int): Frames per second for the gif.
    """
    
    frames = [imageio.imread(frame) for frame in frame_paths]
    out_path = os.path.join(job_path, 'traj.gif')
    imageio.mimsave(out_path, frames, duration = 1000 / fps)



