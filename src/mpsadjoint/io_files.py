import dolfin as df
import dolfin_adjoint as da


def read_active_strain_from_file(U, filename, num_time_steps):
    active_strain = [da.Function(U, name="Active_strain") for _ in range(num_time_steps)]

    with df.XDMFFile(filename) as fin:
        for i, active in enumerate(active_strain):
            fin.read_checkpoint(active, "Active strain", i)

    return active_strain


def write_active_strain_to_file(U, active_strain, filename):
    with df.XDMFFile(filename) as fout:
        for i, active in enumerate(active_strain):
            active_proj = da.project(active, U)
            fout.write_checkpoint(active_proj, "Active strain", i, append=True)


def read_fiber_angle_from_file(U, filename):
    theta = da.Function(U, name="Theta")

    with df.XDMFFile(filename) as fin:
        fin.read_checkpoint(theta, "Theta", 0)

    return theta


def write_fiber_angle_to_file(U, theta, filename):
    theta_proj = df.project(theta, U)

    with df.XDMFFile(filename) as fout:
        fout.write_checkpoint(theta_proj, "Theta", 0)


def write_fiber_direction_to_file(V, theta, filename):
    rotation_matrix = df.as_matrix(
        (
            (df.cos(theta), -df.sin(theta)),
            (df.sin(theta), df.cos(theta)),
        )
    )

    fiber_dir = rotation_matrix * df.as_vector([1.0, 0.0])

    fiber_dir_proj = df.project(fiber_dir, V)

    with df.XDMFFile(filename) as fout:
        fout.write_checkpoint(fiber_dir_proj, "Fiber direction", 0)


def write_states_to_file(states, filename_disp, filename_pressure):
    fout_disp = df.XDMFFile(filename_disp)
    fout_pressure = df.XDMFFile(filename_pressure)

    for i, state in enumerate(states):
        u, p = state.split(deepcopy=True)
        fout_disp.write_checkpoint(u, "Displacement (µm)", i, append=True)
        fout_pressure.write_checkpoint(p, "Pressure (kPa)", i, append=True)

    fout_disp.close()
    fout_pressure.close()


def read_states_from_file(TH, filename_disp, filename_pressure, num_time_steps):
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


def write_displacement_to_file(V, displacements, filename):
    with df.XDMFFile(filename) as fout:
        for i, displacement in enumerate(displacements):
            disp_proj = df.project(displacement, V)
            fout.write_checkpoint(disp_proj, "Displacement (µm)", i, append=True)


def write_strain_to_file(T, strain_values, filename):
    with df.XDMFFile(filename) as fout:
        for i, E in enumerate(strain_values):
            E_proj = df.project(E, T)
            fout.write_checkpoint(E_proj, "Green-Lagrange strain (-)", i, append=True)


def read_displacement_from_file(V, filename, num_time_steps):
    u = [df.Function(V, name="Displacement (µm)") for _ in range(num_time_steps)]

    with df.XDMFFile(filename) as fin:
        for t in range(num_time_steps):
            fin.read_checkpoint(u[t], "Displacement (µm)", t)

    return u
