"""

Åshild Telle / Simula Research Laboratory / 2023

"""


import numpy as np
import dolfin as df
import dolfin_adjoint as da
from scipy.interpolate import RegularGridInterpolator
from types import FunctionType

def mps_to_fenics(mps_data: np.array, mps_info: dict[str, float], mesh: da.Mesh, time_start: int, time_stop: int) -> list[da.Function]:
    """

    Function that maps displacement data, as given per pixel, to a Fenics function space.

    Args:
        mps_data: numpy array of dimensions T x X x Y, where T is the number of time steps,
            X the longitudinal dimension (# blocks), and Y the transverse dimension
        mps_info: dictionary with key information about dimensions in micrometers
        mesh: geometrical domain, fenics mesh
        time_start: time point to read FROM, can be 0 or higher
        time_stop: time point to read TO, can be length of mps_data[0] or lower

    Returns:
        list of displacement data functions, one field per time step

    """

    V2 = df.VectorFunctionSpace(mesh, "CG", 2)

    v_d = V2.dofmap().dofs()
    mesh_coords = V2.tabulate_dof_coordinates()[::2]

    u_data = []

    xcoords, ycoords = define_value_ranges(mps_data, mps_info)

    mesh_xcoords = mesh_coords[:, 0]
    mesh_ycoords = mesh_coords[:, 1]

    for t in range(time_start, time_stop):
        u = da.Function(V2, name=r"Displacement BF data ($\mu m$)")

        ip_fun = mps_interpolation(mps_data[t], xcoords, ycoords)

        xvalues, yvalues = ip_fun(mesh_xcoords, mesh_ycoords)

        u.vector()[v_d] = (
            np.array((xvalues, yvalues)).transpose().flatten()
        )
        u_data.append(u)

    return u_data


def define_value_ranges(mps_data: np.array, mps_info: dict[str, float]) -> tuple[np.array, np.array]:
    """

    Defines x and y coordinates based on info about dimensions and number of points in disp. array.

    Args:
        mps_data: array defining relative displacement values
        mps_info: dictionary with dimension values

    Returns:
        numpy array defining all x values (in µm) per pixel in the longitudinal direction
        numpy array defining all y values (in µm) per pixel in the transverse direction

    """
    um_per_pixel = mps_info["um_per_pixel"]

    _, X, Y, _ = mps_data.shape

    xmax = um_per_pixel / 2 + um_per_pixel * X
    ymax = um_per_pixel / 2 + um_per_pixel * Y

    xvalues = np.linspace(um_per_pixel / 2, xmax, X)
    yvalues = np.linspace(um_per_pixel / 2, ymax, Y)

    return xvalues, yvalues


def mps_interpolation(mps_data: np.array, xvalues: np.array, yvalues: np.array) -> FunctionType:
    """

    Defines an interpolation function, preparing for mapping all mesh points based on disp. values.

    Args:
        mps_data: array defining relative displacement values
        x_values: numpy array defining all x values (in µm) per pixel in the longitudinal direction
        y_values: numpy array defining all y values (in µm) per pixel in the transverse direction
    
    Returns:
        a function: R2 -> R2, taking in x and y coordinates and returning interpolated
            relative displacement for given point along x and y directions

    """
    ip_x = RegularGridInterpolator(
        (xvalues, yvalues),
        mps_data[:, :, 1],
        bounds_error=False,
        fill_value=0,
    )

    ip_y = RegularGridInterpolator(
        (xvalues, yvalues),
        mps_data[:, :, 0],
        bounds_error=False,
        fill_value=0,
    )

    ip_fun = lambda x, y: np.array((ip_x((x, y)), ip_y((x, y))))

    return ip_fun
