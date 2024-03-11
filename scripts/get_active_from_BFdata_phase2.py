"""

Script for doing the first phase of inversion, i.e., with one individual
inversion for all time steps (for each drug dose, if applicable).

This script relies on that the _phase1 script has been run first, and
the following parameters need to match those given for that phase:
    output_folder
    idt
    from_step
    to_step
    step_length
We'll read values in from file based on these parameters.

"""

from argparse import ArgumentParser
import numpy as np

from mpsadjoint.mpsmechanics import mps_to_fenics
from mpsadjoint.mesh_setup import load_mesh_h5
from mpsadjoint.cardiac_mechanics import set_fenics_parameters
from mpsadjoint.inverse import solve_inverse_problem_phase2

set_fenics_parameters()


def parse_cl_arguments():
    parser = ArgumentParser()

    parser.add_argument("--from_step", type=int, default=43, help="Starting time step")
    parser.add_argument("--to_step", type=int, default=44, help="Final time step")

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
        help="How many iterations to use solving the inverse problem in this problem (phase 2)",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help=(
            "Where to save the results (main folder); "
            "if set to None, the results will not be saved."
        ),
    )

    parser.add_argument(
        "--idt",
        type=str,
        default="",
        help="Where to save the results (subfolder name; optional)",
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
        d.num_iterations,
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
    num_iterations,
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
u_data = mps_to_fenics(displacement_data, um_per_pixel, geometry.mesh, from_step, to_step)[
    ::step_length
]

# get initial guesses from phase 1; save output data here after phase 2
init_folder = f"{output_folder}/{study_id}/phase1"
output_folder = f"{output_folder}/{study_id}/phase2"

# solve the inverse problem in phase 2
solve_inverse_problem_phase2(geometry, u_data, num_iterations, init_folder, output_folder)
