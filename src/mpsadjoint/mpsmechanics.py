"""

Åshild Telle / Simula Research Laboratory / 2021

"""


import numpy as np
import dolfin as df
import dolfin_adjoint as da

from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage as nd
from scipy.signal import savgol_filter


def mps_to_fenics(mps_data, mps_info, mesh, time_start, time_stop):
    V2 = df.VectorFunctionSpace(mesh, "CG", 2)

    v_d = V2.dofmap().dofs()
    mesh_coords = V2.tabulate_dof_coordinates()[::2]
    
    u_data = []

    xcoords, ycoords = define_value_ranges(mps_data, mps_info)

    mesh_xcoords = mesh_coords[:, 0]
    mesh_ycoords = mesh_coords[:, 1]

    for t in range(time_start, time_stop):
        u = da.Function(V2, name=r"Displacement BF data ($\mu m$)")

        ip_fun = mps_interpolation(
            mps_data[t],
            xcoords,
            ycoords,
        )

        xvalues, yvalues = ip_fun(mesh_xcoords, mesh_ycoords)

        u.vector()[v_d] = np.array((xvalues, yvalues)).transpose().flatten()
        u_data.append(u)

    return u_data


def define_value_ranges(mps_data, mps_info):

    um_per_pixel = mps_info["um_per_pixel"]

    _, X, Y, _ = mps_data.shape

    xmax = um_per_pixel / 2 + um_per_pixel * X
    ymax = um_per_pixel / 2 + um_per_pixel * Y

    xvalues = np.linspace(um_per_pixel / 2, xmax, X)
    yvalues = np.linspace(um_per_pixel / 2, ymax, Y)

    return xvalues, yvalues


def mps_interpolation(mps_data, xvalues, yvalues):
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
