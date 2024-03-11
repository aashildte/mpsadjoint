"""

Loads mesh + boundary condition surfaces for the pillars (in which Robin BCs will be applied).

Åshild Telle / Simula Research Laboratory / 2024

"""

import os
import typing
import dolfin as df
import dolfin_adjoint as da
from mpi4py import MPI
from pathlib import Path


class Geometry(typing.NamedTuple):
    mesh: df.Mesh
    ds: df.Measure


def load_mesh_h5(
    filename: typing.Union[str, os.PathLike], save_pvd_file: bool = False
) -> Geometry:
    """
    
    Function for loading pre-constructed mesh from h5 file. This mesh should
    match the tissue shape and is expected to have a meshfunction defining
    the pillars.

    Args:
        filename - path to mesh file
        save_pvd_file - if True, the mesh will be saved in a pvd file

    Returns:
        Geometry - nametuple with fields for mesh and pillars 

    """

    comm = MPI.COMM_WORLD
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"File {filename} does not exist")
    if filename.suffix != ".h5":
        raise RuntimeError("File {filename} is not an HDF5 file")

    mesh = da.Mesh()
    with df.HDF5File(comm, filename.as_posix(), "r") as h5_file:
        h5_file.read(mesh, "mesh", False)

        pillar_bcs = df.MeshFunction("size_t", mesh, 1, 0)
        h5_file.read(pillar_bcs, "curves")

    nodes = mesh.num_vertices()
    cells = mesh.num_cells()

    if save_pvd_file:
        df.File("pillar_fun.pvd") << pillar_bcs

    print(f"Number of nodes: {nodes}, number of elements: {cells}")

    ds = df.Measure("ds", domain=mesh, subdomain_data=pillar_bcs)

    return Geometry(mesh=mesh, ds=ds)
