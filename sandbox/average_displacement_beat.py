"""

Finds and computes an average beat from multiple beats.

Åshild Telle, Henrik / Simula Research Laboratory / 2024

"""

import cv2
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import mps
import matplotlib.pyplot as plt


def compute_avg_trace(displacement_data):
    """

    Computes an average of all displacement over space; first by finding
    the norm in each pixel, then by averaging across all X and Y points.

    Args:
        displacement_data: T x X x Y x 2 numpy array

    Returns:
        numpy array of dimension T

    """

    data_normed = da.linalg.norm(
        displacement_data, axis=3
    )  # norm over (x,y)-koordinat -> X x Y x T-shape
    data_avg_in_space = da.mean(data_normed, axis=(1, 2))

    return data_normed

def compute_average_beat(fin: str, reference_time_step: int, pacing_step: int, fout: str, plot_data: bool=False):
    """

    Computes an average displacement beat, pixel by pixel, based on displacement
    data.

    Args:
        input file for displacement: assumed to be an numpy array of dimensions T x X x Y x 2
            where T = time; X, Y = image dimensions; 2 = x and y directions
        reference_time_step: int; manually adjust this to get the best
            resting time step
        pacing_step: int; based on pacing, how many steps per interval
        fout: string, save output values here (should be a npy file)
        plot: boolean, whether or not to plot and display the displacement data

    """

    displacement = da.array(np.load(fin))
    T, X, Y, _ = displacement.shape
    
    num_beats = 0
    all_beats = []

    for index in range(reference_time_step, T - 200, 200):
        all_beats.append(displacement[index : index + 200, :, :])
        num_beats += 1
        print(num_beats)

    avg_beat = da.mean(da.array(all_beats), axis=0)
    
    # save average dispacement beat here
    np.save(
        fout,
        np.array(avg_beat).compute(),
    )

    # plot average trace in order to find reference point, as needed

    if plot_data:
        full_norm = compute_avg_trace(displacement)
        full_time = [dt * t for t in range(0, len(full_norm))]
        
        avg_norm = compute_avg_trace(avg_beat)
        avg_time = [dt * t for t in range(0, len(avg_norm))]

        
        fig, axes = plt.subplots(1, 2, sharey="row", sharex="col", figsize=(7.5, 4.0), gridspec_kw={'width_ratios': [6.0, 1]})
        
        axes[0].plot(full_time, full_norm)
        axes[1].plot(avg_time, avg_norm)
        axes[0].set_ylabel("Displacement (µm)")
        axes[0].set_xlabel("Time (ms)")
        axes[1].set_xlabel("Time (ms)")

        plt.show()


if __name__ == "__main__":
    compute_average_beat(
        "../data/bayK/10nM/displacement_all_beats.npy",
        124,
        200,
        "../data/bayK/10nM/TEST_displacement_avg_beat.npy",
        True
    )
