"""

Script for doing the first phase of inversion, i.e., with one individual
inversion for all time steps AND for all drug doses.

This script relies on that the _phase1+phase2 scripts has been run first, and
the following parameters need to match those given for that phase:
    output_folder
    idt
    from_step
    to_step
    step_length
We'll read values in from file based on these parameters.

"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df

from mpsadjoint.mpsmechanics import mps_to_fenics
from mpsadjoint.mesh_setup import load_mesh_h5
from mpsadjoint.cardiac_mechanics import set_fenics_parameters
from mpsadjoint.inverse import solve_inverse_problem_phase3
from mpsadjoint.io_files import (
    write_displacement_to_file,
    write_strain_to_file,
)

set_fenics_parameters()


def parse_cl_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--from_step", type=int, default=43, help="Starting time step"
    )
    parser.add_argument(
        "--to_step", type=int, default=44, help="Final time step"
    )

    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Stride (consider every _ step)",
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="How many iterations to use solving the inverse problem in this problem (phase 3)",
    )


    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Where to save the results (main folder); if set to None, the results will not be saved.",
    )

    parser.add_argument(
        "--idts",
        nargs="+",
        type=str,
        required=True,
        help="Where to save the results (subfolder names; NOT optional); as a list - length should match displacement_files",
    )

    parser.add_argument(
        "--displacement_files",
        nargs="+",
        type=str,
        required=True,
        help="Displacement data; all files with displacement data over time and space - length should match idts.",
    )

    parser.add_argument(
        "--mesh_file",
        type=str,
        default="",
        required=True,
        help="Mesh file; mesh corresponding to experiment to do the inversion for.",
    )

    parser.add_argument(
        "--um_per_pixel",
        type=float,
        default=1.3552,
        help="Unit conversion parameter; length of each pixel in µm",
    )

    d = parser.parse_args()

    return (
        d.from_step,
        d.to_step,
        d.step_length,
        d.num_iterations,
        d.output_folder,
        d.idts,
        d.displacement_files,
        d.mesh_file,
        d.um_per_pixel,
    )


(
    from_step,
    to_step,
    step_length,
    num_iterations,
    output_folder,
    study_idts,
    displacement_files,
    mesh_file,
    um_per_pixel,
) = parse_cl_arguments()

print(displacement_files)
print(study_idts)
assert len(displacement_files) == len(study_idts), "Error: Please provide one displacement file per study idt."

# load mesh; specific for a given design (including pillar configuration)
geometry = load_mesh_h5(mesh_file)

# convert displacement data to Fenics functions
u_data_all = []
for displacement_file in displacement_files:
    displacement_data = np.load(displacement_file)
    u_data = mps_to_fenics(
        displacement_data, um_per_pixel, geometry.mesh, from_step, to_step
    )[::step_length]
    u_data_all.append(u_data)

# get initial guesses from phase 1; save output data here after phase 2
init_folders = []
output_folders = []

for study_id in study_idts:
    init_folders.append(f"{output_folder}/{study_id}/phase2")
    output_folders.append(f"{output_folder}/{study_id}/phase3")

# solve the inverse problem in phase 3
solve_inverse_problem_phase3(
    geometry, u_data_all, num_iterations, init_folders, output_folders
)
