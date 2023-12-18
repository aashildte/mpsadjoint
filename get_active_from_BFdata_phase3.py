import os
import numpy as np
from argparse import ArgumentParser
# import mps                            # depends on this for getting units

from mpsadjoint import (
    load_mesh_h5,
    solve_inverse_problem_phase_3,
    mps_to_fenics,
    set_fenics_parameters,
)

set_fenics_parameters()


def parse_cl_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--from_step",
        type=int,
        default=243,
        help="Starting time step",
    )
    parser.add_argument(
        "--to_step",
        type=int,
        default=244,
        help="Final time step",
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
        help="How many iterations used to use solving the inverse problem in phase 1",
    )

    parser.add_argument(
        "--num_iterations_phase2",
        type=int,
        default=100,
        help="How many iterations used to use solving the inverse problem in phase 2",
    )

    parser.add_argument(
        "--num_iterations_phase3",
        type=int,
        default=100,
        help="How many iterations to use solving the inverse problem in phase 3",
    )

    parser.add_argument(
        "--mesh_res",
        type=str,
        default="0p5",
        help="mesh resolution",
    )

    d = parser.parse_args()

    return (
        d.from_step,
        d.to_step,
        d.step_length,
        d.num_iterations_phase1,
        d.num_iterations_phase2,
        d.num_iterations_phase3,
        d.mesh_res,
    )


(
    from_step,
    to_step,
    step_length,
    num_iterations_ph1,
    num_iterations_ph2,
    num_iterations_ph3,
    mesh_res,
) = parse_cl_arguments()

# load mesh; specific for a given design
mesh_file = os.path.join("meshes4", f"chip_bayK_clmax_{mesh_res}.h5")
geometry = load_mesh_h5(mesh_file)

folders = []
u_data_all = []
doses = ["Control", "10nM", "100nM", "1000nM"]

for dose in doses:
    displacement_file = f"experiments/AshildData/20211126_bayK_chipB/{dose}/20211126-GCaMP80HCF20-BayK_Stream_B01_s1_TL-20_displacement_avg_beat.npy"

    # load data from mechanical analysis; specific for corresponding experiment
    mps_data = np.load(displacement_file)
    mps_info = {"um_per_pixel": 1.3552}

    u_data = mps_to_fenics(mps_data, mps_info, geometry.mesh, from_step, to_step)[
        ::step_length
    ]
    u_data_all.append(u_data)

    folder = f"new_results/step_{from_step}_{to_step}_{step_length}_bayK_{dose}_msh_{mesh_res}"
    folders.append(folder)


solve_inverse_problem_phase_3(
    geometry,
    u_data_all,
    folders,
    num_iterations_ph1,
    num_iterations_ph2,
    num_iterations_ph3,
)
