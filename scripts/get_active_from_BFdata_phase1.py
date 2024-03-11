"""

Script for doing the first phase of inversion, i.e., with one individual
inversion per time step (and per drug dose, if applicable).

This script runs the process "serially" by using solution from previous
steps as initial guesses to next steps.

Another option, where all steps are individual, is simply to use 
--from_step and --to_step to handle one by one time step, in which case
zero guesses will always be initial guesses.

"""

import os
from argparse import ArgumentParser
import numpy as np
import dolfin as df

from mpsadjoint.mpsmechanics import mps_to_fenics
from mpsadjoint.mesh_setup import load_mesh_h5
from mpsadjoint.cardiac_mechanics import set_fenics_parameters
from mpsadjoint.inverse import solve_inverse_problem_phase1
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
        "--num_iterations_phase1",
        type=int,
        default=100,
        help="How many iterations to use solving the inverse problem (phase 1)",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Where to save the results (main folder); if set to None, the results will not be saved.",
    )

    parser.add_argument(
        "--idt",
        type=str,
        default="study_id",
        help="Where to save the results (study id; this will be a subfolder)",
    )

    parser.add_argument(
        "--displacement_file",
        type=str,
        default="",
        required=True,
        help="Displacement data; file with displacement data over time and space.",
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
        d.num_iterations_phase1,
        d.output_folder,
        d.idt,
        d.displacement_file,
        d.mesh_file,
        d.um_per_pixel,
    )


(
    from_step,
    to_step,
    step_length,
    num_iterations_phase1,
    output_folder,
    study_id,
    displacement_file,
    mesh_file,
    um_per_pixel,
) = parse_cl_arguments()


# load mesh; specific for a given design (including pillar configuration)
geometry = load_mesh_h5(mesh_file)

# convert displacement data to Fenics functions

displacement_data = np.load(displacement_file)
u_data = mps_to_fenics(
    displacement_data, um_per_pixel, geometry.mesh, from_step, to_step
)[::step_length]

# save data here
if study_id != "":
    output_folder += "/" + study_id

output_folder = f"{output_folder}/phase1"

# write original data to file (this is just for comparison; this step can be skipped)
filename_displacement = f"{output_folder}/displacement_original.xdmf"

if not os.path.isfile(filename_displacement):
    V = df.VectorFunctionSpace(geometry.mesh, "CG", 2)
    write_displacement_to_file(V, u_data, filename_displacement)

filename_strain = f"{output_folder}/strain_original.xdmf"
I = df.Identity(2)

E_data = []
for u in u_data:
    F = df.grad(u) + I
    E = 0.5 * (F.T * F - I)
    E_data.append(E)

if not os.path.isfile(filename_strain):
    T2 = df.TensorFunctionSpace(geometry.mesh, "CG", 2)
    write_strain_to_file(T2, E_data, filename_strain)

# solve the inverse problem
solve_inverse_problem_phase1(
    geometry, u_data, num_iterations_phase1, output_folder,
)
