import os
import numpy as np
import dolfin as df
import dolfin_adjoint as da

from mpsadjoint import (
    solve_inverse_problem,
    define_state_space,
    define_bcs,
    define_weak_form,
    set_fenics_parameters,
    write_active_strain_to_file,
    write_fiber_angle_to_file,
    write_fiber_direction_to_file,
    write_displacement_to_file,
    write_strain_to_file,
)
from mpsadjoint.cardiac_mechanics import solve_forward_problem_iteratively
from mpsadjoint.nonlinearproblem import NonlinearProblem


def generate_noise_distribution(original_data, sigma):
    r"""

    Generates a function with a normal distribution $N[0, \sigma]$.

    Args:
        original_data - numpy array (assumed to come from a dolfin function vector)
        sigma - a float; variation in noise to add

    Returns:
        a function in V filled with normally distributed values

    """

    shape = original_data.shape
    noise_fun_values = np.random.normal(loc=original_data, scale=sigma, size=shape)

    return noise_fun_values


def generate_synthetic_data(geometry, active_strain, theta):
    set_fenics_parameters()

    mesh = geometry.mesh
    mesh_finer = df.refine(mesh)

    TH = define_state_space(mesh)
    U = df.FunctionSpace(mesh, "CG", 1)

    active = df.Function(U)
    active.vector()[:] = 0

    theta_fun = df.Function(U)
    theta_fun.assign(df.project(theta, U))

    R, state = define_weak_form(
        TH,
        active,
        theta_fun,
        geometry.ds,
    )

    bcs = define_bcs(TH)
    problem = NonlinearProblem(R, state, bcs=bcs)
    solver = df.NewtonSolver()
    solver.parameters.update(
        {
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
        }
    )

    V2 = df.VectorFunctionSpace(mesh, "CG", 2)
    u_synthetic = []

    for i in range(len(active_strain)):
        print(f"Iterative step: {i} / {len(active_strain)}")
        active_values = df.project(active_strain[i], U)

        solve_forward_problem_iteratively(
            active,
            theta_fun,
            state,
            solver,
            problem,
            active_values,
            theta_fun,
            step_length=1.0,
        )

        _u, _ = state.split()
        u_synthetic.append(df.project(_u, V2))

    return u_synthetic


def inverse_crime(
    geometry,
    active,
    theta,
    output_folder,
    num_iterations_iterative,
    num_iterations_combined,
    noise_level_outer=0,
    noise_level_inner=0,
    mat_params_tissue={},
):

    u_synthetic = generate_synthetic_data(geometry, active, theta)
    zero_dist = np.zeros_like(u_synthetic[0].vector()[:])

    np.random.seed(int(100 * noise_level_outer))

    if noise_level_outer > 0:
        underlying_distribution = generate_noise_distribution(
            zero_dist, noise_level_outer
        )
    else:
        underlying_distribution = zero_dist

    np.save("underlying_distribution.npy", underlying_distribution)

    num_time_steps = len(u_synthetic)
    num_values = len(zero_dist)
    u_synthetic_x = []
    u_synthetic_y = []

    # e1 = df.as_vector([1., 0.])
    # e2 = df.as_vector([0., 1.])

    U = df.FunctionSpace(geometry.mesh, "CG", 2)

    for i in range(num_time_steps):
        u_x = df.project(df.inner(u_synthetic[i], e1), U)
        u_synthetic_x.append(u_x.vector()[:])

        u_y = df.project(df.inner(u_synthetic[i], e2), U)
        u_synthetic_y.append(u_y.vector()[:])

        import IPython

        IPython.embed()
        exit()

    np.save("displacement_x.npy", np.array(u_synthetic_x))
    np.save("displacement_y.npy", np.array(u_synthetic_y))

    V2 = df.VectorFunctionSpace(geometry.mesh, "CG", 2)
    u_data = []

    noise_over_time = np.zeros_like(u_synthetic_x)

    max_fact = 0
    for u in u_synthetic:
        max_fact = max(
            max_fact, df.assemble(df.inner(u, u) * df.dx(geometry.mesh)) ** 0.5
        )

    for i, u in enumerate(u_synthetic):
        u_d = da.Function(V2)

        data_u = u.vector()[:]
        if noise_level_inner > 0:
            noise_distribution_time_step = generate_noise_distribution(
                zero_dist, noise_level_inner
            )
        else:
            noise_distribution_time_step = zero_dist

        scaling_factor = (
            df.assemble(df.inner(u, u) * df.dx(geometry.mesh)) ** 0.5 / max_fact
        )

        noise_fun = da.Function(u.function_space())
        noise_fun.vector()[:] = (
            data_u
            + scaling_factor * noise_distribution_time_step
            + underlying_distribution
        )

        noise_over_time[i, :] = scaling_factor * noise_distribution_time_step

        u_d.vector()[:] = da.project(noise_fun, V2).vector()[:]

        u_data.append(u_d)

    np.save("noise_over_time.npy", noise_over_time)
    exit()

    I = df.Identity(2)
    E_data = []
    for u in u_data:
        F = df.grad(u) + I
        E = 0.5 * (F.T * F - I)
        E_data.append(E)

    write_original_files(output_folder, geometry.mesh, active, theta, u_data, E_data)

    active_m, theta_m, _ = solve_inverse_problem(
        geometry,
        u_data,
        output_folder,
        num_iterations_iterative,
        num_iterations_combined,
        mat_params_tissue,
    )


def write_original_files(output_folder, mesh, active, theta, u_data, E_data):
    U = df.FunctionSpace(mesh, "CG", 1)
    V1 = df.VectorFunctionSpace(mesh, "CG", 1)
    V2 = df.VectorFunctionSpace(mesh, "CG", 2)
    T1 = df.TensorFunctionSpace(mesh, "CG", 1)
    T2 = df.TensorFunctionSpace(mesh, "CG", 2)

    filename_active = f"{output_folder}/active_strain_original.xdmf"
    if not os.path.isfile(filename_active):
        write_active_strain_to_file(U, active, filename_active)

    filename_theta = f"{output_folder}/theta_original.xdmf"
    if not os.path.isfile(filename_theta):
        write_fiber_angle_to_file(U, theta, filename_theta)

    filename_fiber_dir = f"{output_folder}/fiber_direction_original.xdmf"
    if not os.path.isfile(filename_fiber_dir):
        write_fiber_direction_to_file(V1, theta, filename_fiber_dir)

    filename_strain = f"{output_folder}/strain_original_CG1.xdmf"
    if not os.path.isfile(filename_strain):
        write_strain_to_file(T1, E_data, filename_strain)

    filename_displacement = f"{output_folder}/displacement_original.xdmf"
    if not os.path.isfile(filename_displacement):
        write_displacement_to_file(V2, u_data, filename_displacement)
