"""

Files for saving function values to / reading function values from files.
These are used for saving values as output, but also used as a part of
checkpointing in Phase 1.

Åshild Telle / Simula Research Laboratory / 2023

"""

import os
import typing
import ufl_legacy as ufl
import dolfin as df
import dolfin_adjoint as da


def read_active_strain_from_file(
    U: df.FunctionSpace,
    filename: typing.Union[str, os.PathLike],
    num_time_steps: int,
) -> list[da.Function]:
    """

    Creates num_time_steps instances of functions for the active strain
    field, and reads in function values from a given input file.

    Args:
        U - (expected) underlying functionspace
        filename - path to file
        num_time_steps - number of time steps applicable

    Returns:
        list of active strain values, one function per time step

    """

    active_strain = [
        da.Function(U, name="Active strain") for _ in range(num_time_steps)
    ]

    with df.XDMFFile(filename) as fin:
        for i, active in enumerate(active_strain):
            fin.read_checkpoint(active, "Active strain", i)

    return active_strain


def write_active_strain_to_file(
    U: df.FunctionSpace,
    active_strain: list[da.Function],
    filename: typing.Union[str, os.PathLike],
):
    """

    Saves function values for the active strain field to file.

    Args:
        U - underlying functionspace
        active_strain - list of all function instances (one per time step)
        filename - path to saved values

    """
    with df.XDMFFile(filename) as fout:
        for i, active in enumerate(active_strain):
            active_proj = da.project(active, U)
            fout.write_checkpoint(
                active_proj, "Active strain", i, append=True
            )


def read_fiber_angle_from_file(
    U: df.FunctionSpace, filename: typing.Union[str, os.PathLike]
) -> da.Function:
    """

    Creates a function for the fiber angle, and reads function values
    from a given input file.

    Args:
        U - (expected) underlying functionspace
        filename - path to file

    Returns:
        function for the fiber direction angle

    """
    theta = da.Function(U, name="Theta")

    with df.XDMFFile(filename) as fin:
        fin.read_checkpoint(theta, "Theta", 0)

    return theta


def write_fiber_angle_to_file(
    U: df.FunctionSpace,
    theta: da.Function,
    filename: typing.Union[str, os.PathLike],
):
    """

    Saves the fiber angle field to file.

    Args:
        U - underlying functionspace
        theta - function for the fiber direction angle
        filename - path to file

    """
    theta_proj = df.project(theta, U)

    with df.XDMFFile(filename) as fout:
        fout.write_checkpoint(theta_proj, "Theta", 0)


def write_fiber_direction_to_file(
    V: df.FunctionSpace,
    theta: da.Function,
    filename: typing.Union[str, os.PathLike],
):
    """

    Writes the converted fiber direction field to file, i.e., the
    transformation of e_1 to a vector field as determined by the
    underlying fiber direction field.

    Args:
        V - underlying function space (for the vector space)
        theta - function for the fiber direction angle
        filename - path to file

    """

    rotation_matrix = df.as_matrix(
        ((df.cos(theta), -df.sin(theta)), (df.sin(theta), df.cos(theta)))
    )

    fiber_dir = rotation_matrix * df.as_vector([1.0, 0.0])

    fiber_dir_proj = df.project(fiber_dir, V)

    with df.XDMFFile(filename) as fout:
        fout.write_checkpoint(fiber_dir_proj, "Fiber direction", 0)


def write_states_to_file(
    states: list[da.Function],
    filename_disp: typing.Union[str, os.PathLike],
    filename_pressure: typing.Union[str, os.PathLike],
):
    """

    Saves state values to file (as displacement and pressure, separate
    functions).

    Args:
        states - list of states (one per time step)
        filename_disp - path to file
        filename_pressure - path to file

    """

    fout_disp = df.XDMFFile(filename_disp)
    fout_pressure = df.XDMFFile(filename_pressure)

    for i, state in enumerate(states):
        u, p = state.split(deepcopy=True)
        fout_disp.write_checkpoint(u, "Displacement (µm)", i, append=True)
        fout_pressure.write_checkpoint(p, "Pressure (kPa)", i, append=True)

    fout_disp.close()
    fout_pressure.close()


def read_states_from_file(
    TH: df.FunctionSpace,
    filename_disp: typing.Union[str, os.PathLike],
    filename_pressure: typing.Union[str, os.PathLike],
    num_time_steps: int,
) -> list[da.Function]:
    """

    Reads in state values from file (from displacement and pressure
    files, respectively).

    Args:
        TH - function space for the states (typically P2-P1)
        filename_disp - path to file for displacement values
        filename_pressure - path to file for pressure values
        num_time_steps - number of time steps

    returns:
        list of states

    """

    states = [da.Function(TH) for _ in range(num_time_steps)]
    V, U = TH.sub(0).collapse(), TH.sub(1).collapse()

    u = df.Function(V, name="Displacement (µm)")
    p = df.Function(U, name="Pressure (kPa)")

    fin_disp = df.XDMFFile(filename_disp)
    fin_pressure = df.XDMFFile(filename_pressure)

    for i, state in enumerate(states):
        fin_disp.read_checkpoint(u, "Displacement (µm)", i)
        fin_pressure.read_checkpoint(p, "Pressure (kPa)", i)

        df.assign(state.sub(0), u)
        df.assign(state.sub(1), p)

    fin_disp.close()
    fin_pressure.close()

    return states


def write_displacement_to_file(
    V: df.FunctionSpace,
    displacements: list[da.Function],
    filename: typing.Union[str, os.PathLike],
):
    """

    Writes displacement values to file

    Args:
        V - underlying function space
        displacements - list of functions, one per time step
        filename - save here

    """

    with df.XDMFFile(filename) as fout:
        for i, displacement in enumerate(displacements):
            disp_proj = df.project(displacement, V)
            fout.write_checkpoint(
                disp_proj, "Displacement (µm)", i, append=True
            )


def write_strain_to_file(
    T: df.FunctionSpace,
    strain_values: list[ufl.form.Form],
    filename: typing.Union[str, os.PathLike],
):
    """

    Writes strain values to file

    Args:
        T - underlying function space (e.g. CG-2 or DG-2)
        strain_vlaues - list of functions, one per time step
        filename - save here

    """

    with df.XDMFFile(filename) as fout:
        for i, E in enumerate(strain_values):
            E_proj = df.project(E, T)
            fout.write_checkpoint(
                E_proj, "Green-Lagrange strain (-)", i, append=True
            )


def read_displacement_from_file(
    V: df.FunctionSpace,
    filename: typing.Union[str, os.PathLike],
    num_time_steps: int,
) -> list[da.Function]:
    """

    Loads displacement values from file

    Args:
        V - underlying function space
        filename - path to file
        num_time_steps - number of time steps

    Returns:
        list of displacement values - one item per time step

    """

    u = [
        df.Function(V, name="Displacement (µm)")
        for _ in range(num_time_steps)
    ]

    with df.XDMFFile(filename) as fin:
        for t in range(num_time_steps):
            fin.read_checkpoint(u[t], "Displacement (µm)", t)

    return u
