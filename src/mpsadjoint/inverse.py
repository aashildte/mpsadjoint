"""

Code for solving the inverse problem; optimization part. Core part of the code.

Åshild Telle / Simula Research Laboratory / 2024

"""

from __future__ import annotations
from functools import partial
import os
import heapq
import numpy as np
import dolfin as df
import dolfin_adjoint as da

from .nonlinearproblem import NonlinearProblem
from .mesh_setup import Geometry

from .cardiac_mechanics import (
    define_state_space,
    define_bcs,
    define_weak_form,
    solve_forward_problem_iteratively,
)
from .io_files import (
    read_active_strain_from_file,
    read_fiber_angle_from_file,
    read_states_from_file,
    write_active_strain_to_file,
    write_fiber_angle_to_file,
    write_fiber_direction_to_file,
    write_strain_to_file,
    write_states_to_file,
)


class RobustReducedFunctional(da.ReducedFunctional):
    """

    This class overwrites the call function in da.ReducedFunctional by allowing for
    the code to resume if a converged solution has not been found. This step is
    allows the optimization process try out non-converging solutions while maintaining
    the tape used to track previous simulations.

    """

    def __call__(self, *args, **kwargs):
        try:
            value = super().__call__(*args, **kwargs)
        except RuntimeError:
            print(
                "Warning: Forward computation crashed. Resuming...",
                flush=True,
            )
            value = np.nan

        return value


def cost_function(u_model: list[da.Function], u_data: list[da.Function]) -> float:
    r"""

    Computes the difference between u_model and u_data as a surface integral.
    Mathematically, we're calculating

    .. math::
        \sum_{t} \int (u_d (t) - u_m(t), u_d(t) - u_m(t)) dS

    where u_d and u_m are displacement vectors at given time points.

    Args:
        u_model (da.Function): displacement vector function
        u_data (da.Function): displacement vector function

    Returns:
        cost function value (float): total surface integral in x and y components

    """

    surface_int = lambda f: da.assemble(df.inner(f, f) * df.dx)

    cost_fun = 0

    I = df.Identity(2)

    # minimize difference between tracked data and simulated data displacement
    for u_m, u_d in zip(u_model, u_data):
        F_m = I + df.grad(u_m)
        F_d = I + df.grad(u_d)

        E_m = 0.5 * (F_m.T * F_m - I)
        E_d = 0.5 * (F_d.T * F_d - I)

        E_d_int = surface_int(E_d)

        if E_d_int > 0:
            cost_fun += surface_int(E_m - E_d) / E_d_int
        else:
            print("Warning: Strain value found to be zero; omitted from cost function.")

        u_d_int = surface_int(u_d)
        if u_d_int > 0:
            cost_fun += surface_int(u_m - u_d) / u_d_int
        else:
            print("Warning: Displacement value found to be zero; omitted from cost function.")

    print(
        "Cost function displacement difference: ",
        float(cost_fun),
        flush=True,
    )

    return cost_fun


def eval_cb_checkpoint(
    cost_cur: float,
    control_params: list[da.Function],
    tracked_quantities: tuple[list[float], list[list[float]], list[float]],
    mesh: da.Mesh,
):
    """

    This function is called after every successful (acceptable) iteration
    by the inverse solver, tracking certain values of interest.

    Args:
        cost_cur: built-in argument; cost function value
        control_params: built-in argument; states of each control parameter
        tracked_quantities: user-provided; list of values to keep track of
        mesh: mesh used for the FEM simulations

    """

    *active_strain, theta = control_params

    active_tracked = []
    volume = da.assemble(da.Constant(1) * df.dx(mesh))

    for t, active in enumerate(active_strain):
        active_str_t = da.assemble(active * df.dx(mesh)) / volume
        print(
            f"****** Average active strain at time step {t}: {active_str_t:.6f}",
            flush=True,
        )
        active_tracked.append(active_str_t)

    theta_avg = da.assemble(theta * df.dx(mesh)) / volume
    tracked_quantities[0].append(cost_cur)
    tracked_quantities[1].append(active_tracked)
    tracked_quantities[2].append(theta_avg)


def initiate_controls(
    mesh: da.Mesh,
    init_active_strain: list[da.Function] | list[da.Constant],
    init_theta: da.Function | da.Constant,
) -> tuple[list[da.Function], da.Function]:
    """

    Initiates variables for active strain + theta, given initial guesses.

    Args:
        mesh - computational domain, mesh representation
        init_active_strain: initial values (can be constants)
            for active strain values
        init_theta: initial values (can be constant) for the
            fiber direction angle

    Returns:
        active_strain: list representing the active strain,
            for each time step
        theta: control representing the fiber angle,
            constant for all time steps

    """
    U = df.FunctionSpace(mesh, "CG", 1)

    active_strain = []

    for init_active_str in init_active_strain:
        active_str = da.Function(U, name="Active stress")
        active_str.assign(df.project(init_active_str, U))
        active_strain.append(active_str)

    theta = da.Function(U, name="Theta")
    theta.assign(df.project(init_theta, U))

    return active_strain, theta


def define_forward_problems(
    geometry: Geometry,
    active_strain: list[da.Function],
    theta: da.Function,
    init_states: list[da.Function] | None = None,
) -> tuple[list[da.NewtonSolver], list[NonlinearProblem], list[da.Function]]:
    """

    Defines all forward (mechanical) problems; including interpolation
    based on previous states when applicable.

    Args:
        geometry: Geometry object; mesh + ds defined over pillars
        active_strain: list of active strain functions
        theta: fiber direction angle function
        init_states: start from here; initial guesses. Not used if not set.

    Returns:
        newtonsolver used to solve the forward problems
        forward_problems: corresponding nonlinear problems, for all time steps
        states: state for all time steps

    """

    TH = define_state_space(geometry.mesh)
    bcs = define_bcs(TH)

    newtonsolver = da.NewtonSolver()
    newtonsolver.parameters.update({"relative_tolerance": 1e-5, "absolute_tolerance": 1e-5})

    forward_problems = []
    if init_states is None:
        init_states = []
    states = []

    for time_step, active_str in enumerate(active_strain):
        print("Solving forward problem step ", time_step, flush=True)
        weak_form, state = define_weak_form(TH, active_str, theta, geometry.ds)

        if len(init_states) > time_step:
            state.vector()[:] = init_states[time_step].vector()[:]

        forward_problem = NonlinearProblem(weak_form, state, bcs=bcs)
        newtonsolver.solve(forward_problem, state.vector())

        forward_problems.append(forward_problem)
        states.append(state)

    return newtonsolver, forward_problems, states


def define_optimization_problem(
    states: list[da.Function],
    u_data: list[da.Function],
    mesh: da.Mesh,
    control_variables: list[da.Function],
) -> tuple[da.MinimizationProblem, tuple[list[float], list[list[float]], list[float]]]:
    """
    Defines key variables for the optimization problem, as well as the
    problem itself.

    Args:
        states: list of da.Function objects, all states over time
        u_data: list of da.Function objects, data displacement over time
        mesh: geometrical domain
        control variables: list of da.Function objects, to be optimized

    Returns:
        optimization (minimization) problem
        a lists of cost function and average values, to be
            updated after every acceptable successful iteration

    """
    u_model = [state.split()[0] for state in states]

    cost_fun = cost_function(u_model, u_data)

    control_params = [da.Control(x) for x in control_variables]

    # track 3 values : cost fun, active, theta
    tracked_quantities: tuple[list[float], list[list[float]], list[float]] = (
        [],
        [],
        [],
    )

    eval_cb = partial(
        eval_cb_checkpoint,
        tracked_quantities=tracked_quantities,
        mesh=mesh,
    )

    reduced_functional = RobustReducedFunctional(cost_fun, control_params, eval_cb_post=eval_cb)

    # one bound for every active tension field + one bound for the fiber dir. angle
    bounds = [(0, 0.3)] * (len(control_variables) - 1) + [(-np.pi / 2, np.pi / 2)]

    optimization_problem = da.MinimizationProblem(reduced_functional, bounds=bounds)

    return optimization_problem, tracked_quantities


def solve_optimization_problem(
    problem: da.MinimizationProblem, maximum_iterations: int
) -> list[da.Function]:
    """

    Initiates and runs optimization problem through IPOPT.

    Args:
        problem: minimization problem to solve
        maximum_iterations: number of iterations to
            run the optimization algorithmn for

    Returns:
        list of optimal values found

    """

    parameters = {
        "limited_memory_initialization": "scalar2",
        "maximum_iterations": maximum_iterations,
        "mu_strategy": "adaptive",
        "adaptive_mu_globalization": "never-monotone-mode",
        "sigma_max": 10.0,
        "alpha_red_factor": 0.05,
    }

    inv_solver = da.IPOPTSolver(problem, parameters=parameters)
    optimal_values = inv_solver.solve()

    return optimal_values


def write_inversion_results(
    mesh: da.Mesh,
    active: list[da.Function],
    theta: da.Function,
    states: list[da.Function],
    output_folder: str,
):
    """
    Saves results to disk by saving all as dolfin functions (xdmf files).
    Some of these are also used as checkpoint values in case the
    optimization process is disrupted.

    Args:
        mesh: geometrical domain
        active: list of active tension fields (one function per time step)
        theta: fiber direction field
        states: list of states (P2-P1 functions; one state per time step)
        output_folder: save output values here

    """

    U = df.FunctionSpace(mesh, "CG", 1)
    V1 = df.VectorFunctionSpace(mesh, "CG", 1)
    T1 = df.TensorFunctionSpace(mesh, "CG", 1)
    T2 = df.TensorFunctionSpace(mesh, "CG", 2)

    filename_active = f"{output_folder}/active_strain.xdmf"
    write_active_strain_to_file(U, active, filename_active)

    filename_fiber_dir = f"{output_folder}/fiber_dir.xdmf"
    write_fiber_direction_to_file(V1, theta, filename_fiber_dir)

    filename_theta = f"{output_folder}/theta.xdmf"
    write_fiber_angle_to_file(U, theta, filename_theta)

    filename_disp = f"{output_folder}/displacement.xdmf"
    filename_pressure = f"{output_folder}/pressure.xdmf"
    write_states_to_file(states, filename_disp, filename_pressure)

    filename_strain_CG1 = f"{output_folder}/strain_CG1.xdmf"
    filename_strain_CG2 = f"{output_folder}/strain_CG2.xdmf"

    I = df.Identity(2)
    strain_values = []
    for s in states:
        u, _ = s.split()
        F = I + df.grad(u)
        E = 0.5 * (F.T * F - I)
        strain_values.append(E)

    write_strain_to_file(T1, strain_values, filename_strain_CG1)
    write_strain_to_file(T2, strain_values, filename_strain_CG2)


def write_inversion_statistics(
    tracked_quantities: tuple[list[float], list[list[float]], list[float]],
    output_folder: str,
):
    """
    Saves cost function values + average control values to file.
    """

    cost_function_values = np.array(tracked_quantities[0])
    active_strain = np.array(tracked_quantities[1])
    theta = np.array(tracked_quantities[2])

    np.save(f"{output_folder}/cost_function.npy", cost_function_values)
    np.save(f"{output_folder}/active_strain.npy", active_strain)
    np.save(f"{output_folder}/theta.npy", theta)


def sort_data_after_displacement(
    u_data: list[da.Function],
) -> list[da.Function]:
    """
    Sort the data, i.e. actually the time steps used to access the data, after
    displacement norm

    Args:
        u_data: list of displacement functions representing original data

    Returns:
        heap with displacement norms as value, time_steps as values

    """

    num_time_steps = len(u_data)
    heap: list[tuple[float, int]] = []

    for time_step in range(num_time_steps):
        disp_norm = df.assemble(df.inner(u_data[time_step], u_data[time_step]) * df.dx)
        heapq.heappush(heap, (disp_norm, time_step))

    return heap


def solve_pdeconstrained_optimization_problem(
    geometry: Geometry,
    u_data: list[da.Function],
    init_active_strain: list[da.Function] | list[da.Constant],
    init_theta: da.Function | da.Constant,
    init_states: list[da.Function],
    number_of_iterations: int,
) -> tuple[
    list[da.Function],
    da.Function,
    list[da.Function],
    tuple[list[float], list[list[float]], list[float]],
]:
    """

    Args:
        geometry: mesh + ds for pillars
        u_data: list of displacement functions representing original data
        init_active_strain: list of initial guesses for the active strain field
        init_theta: initial guess for the fiber direction angle function
        init_states: list of states corresponding to the solution of the
            init_active_strain and init_theta fields
        number_of_iterations: number of iterations to use in the optimization problem

    Returns:
        list of active strain fields, one per time step
        fiber direction angle field
        list of corresponding states (disp. + pressure)
        list of values of interest tracked in the optimization process

    """

    active_strain, theta = initiate_controls(geometry.mesh, init_active_strain, init_theta)

    newtonsolver, forward_problems, states = define_forward_problems(
        geometry, active_strain, theta, init_states
    )

    control_variables = active_strain[:] + [theta]

    problem, tracked_quantities = define_optimization_problem(
        states, u_data, geometry.mesh, control_variables
    )

    optimal_values = solve_optimization_problem(problem, number_of_iterations)

    *optimal_active_strain, optimal_theta = optimal_values

    for active, state, optimal_active, problem in zip(
        active_strain, states, optimal_active_strain, forward_problems
    ):
        solve_forward_problem_iteratively(
            active,
            theta,
            state,
            newtonsolver,
            problem,
            optimal_active,
            optimal_theta,
        )

    return active_strain, theta, states, tracked_quantities


def data_exist(output_folder: str) -> bool:
    """

    Checks whether we already have calculated optimized values; this
    is a part of the checkpointing process. If all of active strain,
    theta, and displacement exists on disk, this function will return
    True.

    Args:
        output_folder: folder in which the output values might exist

    Returns:
        bool: True if all three files exist

    """

    active = os.path.isfile(output_folder + "/active_strain.xdmf")
    theta = os.path.isfile(output_folder + "/theta.xdmf")
    states = os.path.isfile(output_folder + "/displacement.xdmf")

    return active and theta and states


def load_data(output_folder, U, TH, num_time_steps):
    active = read_active_strain_from_file(U, output_folder + "/active_strain.xdmf", num_time_steps)
    theta = read_fiber_angle_from_file(U, output_folder + "/theta.xdmf")
    states = read_states_from_file(
        TH,
        output_folder + "/displacement.xdmf",
        output_folder + "/pressure.xdmf",
        num_time_steps,
    )

    return active, theta, states


def solve_inverse_problem_phase1(
    geometry: Geometry,
    u_data: list[da.Function],
    number_of_iterations: int = 100,
    output_folder: str | None = None,
) -> tuple[list[da.Function], list[da.Function], list[da.Function]]:
    """

    Solves the inverse problem for phase 1, i.e., for each time step
    and for each drug dose, as applicable. This is done in order of magnitude,
    as in, we first sort the time steps according to magntiude of displacement,
    and then use the solution of each of those as input to the next in order.

    Values are written to file (useful for checkpointing) as weil as returned
    (useful for running a combined script).

    Args:
        geometry: mesh + ds for pillars
        u_data: list of functions for original displacement
        number_of_iterations: number of iterations for the min. problem
        output_folder: save results here

    Returns:
        list of active strain fields, per time step
        fiber direction angle field
        corresponding states (disp. and pressure)

    """

    num_time_steps = len(u_data)
    heap = sort_data_after_displacement(u_data)

    # initial values; to be replaced as we progress more time steps
    active_strain = [da.Constant(0.0)]
    theta = da.Constant(0.0)
    states = []

    active_strain_over_time = [None] * num_time_steps
    theta_over_time = [None] * num_time_steps
    states_over_time = [None] * num_time_steps

    TH = define_state_space(geometry.mesh)
    U = df.FunctionSpace(geometry.mesh, "CG", 1)

    while heap:
        _, time_step = heapq.heappop(heap)

        if output_folder is not None:
            output_folder_t = f"{output_folder}/iteration_{time_step}"

        iteration_num = num_time_steps - len(heap)

        if output_folder is not None and data_exist(output_folder_t):
            print(
                f"Loading inversion results for time step {time_step}; "
                f"iteration {iteration_num} / {len(u_data)}",
                flush=True,
            )
            active_strain, theta, states = load_data(output_folder_t, U, TH, 1)
        else:
            print(
                f"Starting inversion for time step {time_step}; "
                f"iteration {iteration_num} / {len(u_data)}",
                flush=True,
            )
            (
                active_strain,
                theta,
                states,
                tracked_quantities,
            ) = solve_pdeconstrained_optimization_problem(
                geometry,
                [u_data[time_step]],
                active_strain,
                theta,
                states,
                number_of_iterations,
            )

            if output_folder is not None:
                write_inversion_results(
                    geometry.mesh,
                    active_strain,
                    theta,
                    states,
                    output_folder_t,
                )

                write_inversion_statistics(tracked_quantities, output_folder_t)

        theta_over_time[time_step] = da.project(theta, U)
        active_strain_over_time[time_step] = da.project(active_strain[0], U)
        states_over_time[time_step] = states[0]

    return active_strain_over_time, theta_over_time, states_over_time


def solve_inverse_problem_phase2(
    geometry: Geometry,
    u_data: list[da.Function],
    number_of_iterations: int = 100,
    input_folder: str | None = None,
    output_folder: str | None = None,
) -> tuple[list[da.Function], da.Function, list[da.Function]]:
    """

    Main function; we'll run everything from here.

    Args:
        geometry: namedtuple Geometry with mesh and surface information
        u_data: list of displacement data for all time points
        number_of_iteration: integer; number of iterations to use
            for optimizing all time steps combined into one optimization problem
        input_folder: results from phase 1 are assumed to be saved here;
            if None, we will go straight into phase 2 (zero-valued
            initial guesses assumed)
        output_folder: results will be saved here (if provided)

    Returns:
        active_strain: list of da.Function objects, contains optimized values
            for active strain, for each time step
        theta: da.function object, contains optimized values for the fiber
            direction (as an angle), for all time steps altogether
        states: list of corresponding states (disp. + pressure), solutions
            based on active strain and theta values

    """

    active_over_time = []
    theta_over_time = []
    states_over_time = []

    TH = define_state_space(geometry.mesh)
    U = df.FunctionSpace(geometry.mesh, "CG", 1)

    if input_folder is not None:
        for i in range(len(u_data)):
            data_it = input_folder + f"/iteration_{i}"
            assert data_exist(data_it), "Error: Input folder provided but data does not exist."

            active, theta, state = load_data(data_it, U, TH, len(u_data))

            active_over_time.append(active[0])
            theta_over_time.append(theta)
            states_over_time.append(state[0])

        # use average value as initial guess for phase 2

        theta = df.Function(U, name="Theta")
        theta_sum = 0
        for th in theta_over_time:
            theta_sum += theta

        theta.assign(df.project(theta_sum / len(theta_over_time), U))

    else:
        num_time_step = len(u_data)
        active_over_time = [da.Function(U, name="Active strain")] * num_time_step
        theta = da.Function(U, name="Theta")
        states = [da.Function(TH)] * num_time_step

    states_over_time2 = []

    # solve forward problems: make sure initial guesses will converge

    for t, (active, theta_t, state) in enumerate(
        zip(active_over_time, theta_over_time, states_over_time)
    ):
        print("Solving for time step ", t, flush=True)
        newtonsolver, forward_problems, state2 = define_forward_problems(
            geometry, [active], theta_t, init_states=[state]
        )

        solve_forward_problem_iteratively(
            active,
            theta_t,
            state2[0],
            newtonsolver,
            forward_problems[0],
            active,
            theta,
            step_length=1.0,
        )
        states_over_time2.append(state)

    (
        active_strain,
        theta,
        states,
        tracked_quantities,
    ) = solve_pdeconstrained_optimization_problem(
        geometry,
        u_data,
        active_over_time,
        theta,
        states_over_time2,
        number_of_iterations,
    )

    if output_folder is not None:
        write_inversion_results(geometry.mesh, active_strain, theta, states, output_folder)

        write_inversion_statistics(tracked_quantities, output_folder)

    return active_strain, theta, states


def solve_inverse_problem_phase3(
    geometry: Geometry,
    u_data: list[da.Function],
    number_of_iterations: int = 100,
    input_folders: list[str] | None = None,
    output_folders: list[str] | None = None,
) -> tuple[list[da.Function], da.Function, list[da.Function]]:
    """

    Main function for the third phase inversion.

    Args:
        geometry: namedtuple Geometry with mesh and surface information
        u_data: list of lists of displacement data for all time points
            (so one list per drug dose)
        number_of_iteration: integer; number of iterations to use
            for optimizing all time steps combined into one optimization problem
        input_folders: results from phase 1 are assumed to be saved here,
            in a folder structure matching the folder structure for u_data
        output_folder: results will be saved here (if provided),
            in a folder structure matching the folder structure for u_data

    Returns:
        active_strain: list of da.Function objects, contains optimized values
            for active strain, for each time step
        theta: da.function object, contains optimized values for the fiber
            direction (as an angle), for all time steps altogether
        states: list of corresponding states (disp. + pressure), solutions
            based on active strain and theta values

    """

    active_all = []
    states_all = []
    u_data_all = []
    theta_all = []

    TH = define_state_space(geometry.mesh)
    U = df.FunctionSpace(geometry.mesh, "CG", 1)

    T = len(u_data[0])

    if input_folders is not None:
        for folder in input_folders:
            assert data_exist(folder), f"Error: Data in folder {folder} do not exist."
            print(folder, len(u_data[0]))
            active_strain, theta, states = load_data(folder, U, TH, len(u_data[0]))

            active_all += active_strain
            states_all += states
            theta_all.append(theta)
    else:
        N = len(u_data) * len(u_data[0])
        # active_over_time = [da.Function(U, name="Active strain")] * N
        states = [da.Function(TH)] * N
        theta = da.Function(U, name="Theta")

    for u_d in u_data:
        # this is technically not so important, but the code assumes this to be true:
        assert len(u_d) == len(
            u_data[0]
        ), "Error: number of time steps per dose needs to be the same!"
        u_data_all += u_d

    assert len(active_all) == len(states_all) and len(active_all) == len(
        u_data_all
    ), "Error: Active / states / data do not have the same length."

    # define average theta value from the different drug doses as an initial guess
    # to the final optimization phase
    theta_avg = df.Function(U, name="Theta")

    theta_sum = 0
    for theta in theta_all:
        theta_sum += theta

    theta_avg.assign(df.project(theta_sum / len(theta_all), U))
    # syncronizing theta values iterative (recursively for convergence)

    states_all_syncronized = []

    for i, theta_t in enumerate(theta_all):
        print("Dose: ", i)
        for t, (active, state) in enumerate(
            zip(
                active_all[i * T : (i + 1) * T],
                states_all[i * T : (i + 1) * T],
            )
        ):
            print("Solving for time step ", t, flush=True)
            (
                newtonsolver,
                forward_problems,
                state2,
            ) = define_forward_problems(geometry, [active], theta_t, init_states=[state])

            solve_forward_problem_iteratively(
                active,
                theta_t,
                state2[0],
                newtonsolver,
                forward_problems[0],
                active,
                theta_avg,
                step_length=1.0,
            )
            states_all_syncronized.append(state)

    (
        active_all,
        theta,
        states,
        tracked_quantities,
    ) = solve_pdeconstrained_optimization_problem(
        geometry,
        u_data_all,
        active_all,
        theta_avg,
        states_all_syncronized,
        number_of_iterations,
    )

    if output_folders is None:
        output_folders = []

    for i, output_folder in enumerate(output_folders):
        active_strain_per_dose = active_all[i * T : (i + 1) * T]
        states_per_dose = states[i * T : (i + 1) * T]

        write_inversion_results(
            geometry.mesh,
            active_strain_per_dose,
            theta,
            states_per_dose,
            output_folder,
        )

        write_inversion_statistics(tracked_quantities, output_folder)

    return active_all, theta, states_all
