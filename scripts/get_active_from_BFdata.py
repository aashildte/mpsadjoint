import os
import numpy as np
import dolfin as df
from argparse import ArgumentParser
import mps  # depends on this for getting units

from mpsadjoint.mpsmechanics import mps_to_fenics
from mpsadjoint.mesh_setup import load_mesh_h5
from mpsadjoint.cardiac_mechanics import set_fenics_parameters
from mpsadjoint.io_files import write_displacement_to_file, write_strain_to_file

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
        "--num_iterations_iterative",
        type=int,
        default=100,
        help="How many iterations to use solving the inverse problem",
    )

    parser.add_argument(
        "--num_iterations_combined",
        type=int,
        default=100,
        help="How many iterations to use solving the inverse problem",
    )

    parser.add_argument("--dose", type=str, default="Control", help="dose (folder)")

    parser.add_argument(
        "--mesh_res",
        type=str,
        default="20",
        help="mesh resolution",
    )

    d = parser.parse_args()
    return (
        d.from_step,
        d.to_step,
        d.step_length,
        d.num_iterations_iterative,
        d.num_iterations_combined,
        d.dose,
        d.mesh_res,
    )


(
    from_step,
    to_step,
    step_length,
    num_iterations_iterative,
    num_iterations_combined,
    dose,
    mesh_res,
) = parse_cl_arguments()

displacement_file = (
    "experiments/AshildData/20211126_bayK_chipB/"
    f"{dose}/20211126-GCaMP80HCF20-BayK_Stream_B01_s1_TL-20_displacement_avg_beat.npy"
)
mps_file = (
    "experiments/AshildData/20220105_omecamtiv_chipB/"
    f"{dose}/20220105-80GCaMP20HCF-omecamtiv_Stream_B01_s1_TL-20-Stream.tif"
)
# load data from mechanical analysis; specific for corresponding experiment
mps_data = np.load(displacement_file)
mps_info = mps.MPS(mps_file).info
mps_info = {"um_per_pixel": 1.3552}

# then get command line arguments

output_folder = f"new_results/step_{from_step}_{to_step}_{step_length}_bayK_{dose}_msh_{mesh_res}"

# load mesh; specific for a given design
mesh_file = f"meshes4/chip_bayK_clmax_{mesh_res}.h5"
geometry = load_mesh_h5(mesh_file)

u_data = mps_to_fenics(mps_data, mps_info, geometry.mesh, from_step, to_step)[::step_length]

filename_displacement = f"{output_folder}/displacement_original.xdmf"

if not os.path.isfile(filename_displacement):
    V = df.VectorFunctionSpace(geometry.mesh, "CG", 2)
    write_displacement_to_file(V, u_data, filename_displacement)

filename_strain_CG1 = f"{output_folder}/strain_original_CG1.xdmf"
filename_strain_CG2 = f"{output_folder}/strain_original_CG2.xdmf"

I = df.Identity(2)

E_data = []
for u in u_data:
    F = df.grad(u) + I
    E = 0.5 * (F.T * F - I)
    E_data.append(E)

if not os.path.isfile(filename_strain_CG1):
    T1 = df.TensorFunctionSpace(geometry.mesh, "CG", 1)
    write_strain_to_file(T1, E_data, filename_strain_CG1)

if not os.path.isfile(filename_strain_CG2):
    T2 = df.TensorFunctionSpace(geometry.mesh, "CG", 2)
    write_strain_to_file(T2, E_data, filename_strain_CG2)

# solve_inverse_problem(
#     geometry,
#     u_data,
#     output_folder,
#     num_iterations_iterative,
#     num_iterations_combined,
# )
