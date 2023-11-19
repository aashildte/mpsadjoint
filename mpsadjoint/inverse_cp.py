from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import dolfin as df
import dolfin_adjoint as da

from mpsadjoint import (
    def_state_space,
    def_bcs,
    def_weak_form,
    solve_forward_problem_iteratively,
)

from mpsadjoint.nonlinearproblem import NonlinearProblem

class RobustReducedFunctional(da.ReducedFunctional):
    def __call__(self, *args, **kwargs):
        try:
            value = super().__call__(*args, **kwargs)
        except RuntimeError:
            print("Warning: Forward computation crashed. Resuming...", flush=True)
            value = np.nan

        return value

def cost_diff(mesh, facets, u_model, u_data):
    cost_fun = 0

    e1 = df.as_vector([1.0, 0.0, 0.0])
    e2 = df.as_vector([0.0, 1.0, 0.0])

    ds = df.Measure("ds", domain=mesh, subdomain_data=facets)
    area = da.assemble(df.inner(1, 1) * ds(1))
    surf_int = lambda f: da.assemble(df.inner(f, f) * ds(1)) / area

    # minimize difference between tracked data and simulated data displacement
    for (u_m, u_d) in zip(u_model, u_data):
        cost_fun += surf_int(df.inner(u_m, e1) - df.inner(u_d, e1))  # x component
        cost_fun += surf_int(df.inner(u_m, e2) - df.inner(u_d, e2))  # y component

    print("cost function disp: ", float(cost_fun))

    return cost_fun


def spatial_reg(mesh, active_strain):
    # regularize spatial derivative
    cost_fun = 0
    volume = da.assemble(df.inner(1, 1) * df.dx(mesh))

    zdir = df.as_vector([0, 0, 1])

    # regularize spatial derivative
    for cntr in active_strain:
        grad_contr = df.grad(cntr)
        z_grad = df.inner(grad_contr, zdir)
        cost_fun += da.assemble(df.inner(z_grad, z_grad) * df.dx) / volume

    return cost_fun


def cost_function(
    u_model,
    u_data,
    controls,
    mesh,
    facets,
    reg_alpha=1e5,
):
    cf = cost_diff(mesh, facets, u_model, u_data)

    # assume here that all or none of the controls are constants
    if not isinstance(controls[0], da.Constant):
        cf += reg_alpha * spatial_reg(mesh, controls)

    return cf


def initiate_controls_continuous(U, init_active, init_theta):

    num_time_steps = len(init_active)
    active_strain = [
        da.Function(U, name="Active stress") for _ in range(num_time_steps)
    ]

    for (a, i) in zip(active_strain, init_active):
        a.assign(df.project(i, U))

    theta = da.Function(U, name="Theta")
    theta.assign(df.project(init_theta, U))

    return active_strain, theta


def initiate_controls_constant(init_active):
    return [da.Constant(ia) for ia in init_active]


def eval_cb_checkpoint(
    cost_cur,
    control_params,
    tracked_values,
):
    tracked_values.append(cost_cur)


def active_to_scaled(active):
    return active / da.Constant(0.3)


def theta_to_scaled(theta):
    return (theta + da.Constant(np.pi / 2)) / da.Constant(np.pi)


def scaled_to_active(active_scaled):
    return da.Constant(0.3) * active_scaled


def scaled_to_theta(theta_scaled):
    return da.Constant(-np.pi / 2) + da.Constant(np.pi) * theta_scaled


def define_control_variables(init_active, init_theta, constant_controls, U):
    init_active_scaled = [active_to_scaled(ia) for ia in init_active]
    init_theta_scaled = theta_to_scaled(init_theta)

    # TODO let init_controls_constant return theta as well

    if constant_controls:
        active_strain_scaled = initiate_controls_constant(init_active_scaled)
        theta = da.Constant(0.5)

    else:
        active_strain_scaled, theta_scaled = initiate_controls_continuous(
            U, init_active_scaled, init_theta_scaled
        )

        theta = scaled_to_theta(theta_scaled)

    active_strain = []

    for act_sc in active_strain_scaled:
        act = scaled_to_active(act_sc)
        active_strain.append(act)

    if constant_controls:
        control_variables = active_strain_scaled[:]
    else:
        control_variables = active_strain_scaled[:] + [theta_scaled]

    return control_variables, active_strain, theta


def define_forward_problems(active_strain, theta, init_states, TH, bcs):
    
    # TODO consider moving this to cardiac_mechanics

    newtonsolver = da.NewtonSolver()
    newtonsolver.parameters["absolute_tolerance"] = 1e-5

    forward_problems = []
    states = []
    u_model = []

    num_time_steps = len(active_strain)

    for t in range(num_time_steps):
        R, state = def_weak_form(
            TH,
            active_strain[t],
            theta,
        )
        forward_problem = NonlinearProblem(R, state, bcs)

        state.vector()[:] = init_states[t].vector()[:]
        newtonsolver.solve(forward_problem, state.vector())

        forward_problems.append(forward_problem)
        states.append(state)

        u, _ = df.split(state)
        u_model.append(u)

    return newtonsolver, forward_problems, states, u_model


def define_reduced_functional(u_model, u_data, mesh, ceiling, control_variables):

    cost_fun = cost_function(
        u_model,
        u_data,
        control_variables,
        mesh,
        ceiling,
    )

    control_params = [da.Control(x) for x in control_variables]

    cost_over_iters = []
    eval_cb = partial(
        eval_cb_checkpoint,
        tracked_values=cost_over_iters,
    )

    reduced_functional = RobustReducedFunctional(
        cost_fun, control_params, eval_cb_post=eval_cb
    )

    return reduced_functional, cost_over_iters


def run_inverse_solver(
    problem, control_params, iterations, newtonsolver, forward_problems, states
):
    parameters = {
        "limited_memory_initialization": "scalar2",
        "maximum_iterations": iterations,
        "alpha_red_factor": 0.2,
        "bound_relax_factor": 0.02,
        "honor_original_bounds": "yes",
        "max_soc": 0,
        "bound_mult_init_val": 0.01,
        "mu_strategy": "adaptive",
        "adaptive_mu_globalization": "never-monotone-mode",
        "sigma_max": 10.0,
        "quality_function_norm_type": "max-norm",
        "theta_max_fact": 0.02,
    }

    inv_solver = da.IPOPTSolver(problem, parameters=parameters)
    optimal_values = inv_solver.solve()

    return optimal_values

def solve_inverse_problem_inner(
    mesh,
    ceiling,
    u_data,
    init_active,
    init_theta,
    init_states,
    TH,
    U,
    bcs,
    constant_controls,
    iters_inner,
):
    
    control_variables, active_strain, theta = define_control_variables(
        init_active, init_theta, constant_controls, U
    )
    newtonsolver, forward_problems, states, u_model = define_forward_problems(
        active_strain, theta, init_states, TH, bcs
    )
    
    reduced_functional, cost_over_iters = define_reduced_functional(
        u_data, u_model, mesh, ceiling, control_variables
    )

    bounds = [(0, 1)] * len(control_variables)
    problem = da.MinimizationProblem(reduced_functional, bounds=bounds)
    optimal_values = run_inverse_solver(
        problem, control_variables, iters_inner, newtonsolver, forward_problems, states
    )

    *active_strain_scaled, theta_scaled = control_variables
    *optimal_active_strain, optimal_theta = optimal_values

    for active_scaled, state, optimal_active, init_state, problem in zip(
            active_strain_scaled, states, optimal_active_strain, init_states, forward_problems
    ):
        
        solve_forward_problem_iteratively(
            active_scaled,
            theta_scaled,
            state,
            newtonsolver,
            problem,
            optimal_active,
            optimal_theta,
            init_state,
        )
    return (
        active_strain,
        theta,
        states,
        cost_over_iters,
    )


def write_results(U, V, u_data, active, theta, states, cost_over_outer, output_folder):

    disp_org = df.XDMFFile(f"{output_folder}/disp_org.xdmf")
    disp_est = df.XDMFFile(f"{output_folder}/disp_est.xdmf")
    active_est = df.XDMFFile(f"{output_folder}/active_strain.xdmf")

    for i, (u_org, a, s) in enumerate(zip(u_data, active, states)):
        u_df = df.project(u_org, V)
        disp_org.write_checkpoint(u_df, "Displacement (µm)", i, append=True)

        a_df = df.project(a, U)
        active_est.write_checkpoint(a_df, "Active strain", i, append=True)

        u, _ = s.split()
        u_df = df.project(u, V)
        disp_est.write_checkpoint(u_df, "Displacement (µm)", i, append=True)

    theta = da.project(theta, U)

    rot_matrix = df.as_matrix(
        (
            (df.cos(theta), da.Constant(-1) * df.sin(theta), 0),
            (df.sin(theta), df.cos(theta), 0),
            (0, 0, 1),
        )
    )
    fiber_dir = rot_matrix * df.as_vector([1.0, 0.0, 0.0])
    u_df.assign(df.project(fiber_dir, V))

    dir_file = df.XDMFFile(f"{output_folder}/fiber_dir.xdmf")
    dir_file.write_checkpoint(u_df, "Fiber dir.", 0)
    dir_file.close()

    theta_file = df.XDMFFile(f"{output_folder}/theta.xdmf")
    theta_file.write_checkpoint(theta, "theta", 0)
    theta_file.close()

    disp_est.close()
    disp_org.close()
    active_est.close()

    np.save(f"{output_folder}/cost_function.npy", np.array(cost_over_outer))

    # then plot
    plt.plot(cost_over_outer)
    plt.xlabel("Iteration")
    plt.ylabel("Cost function")
    # plt.yscale("log")
    plt.savefig(f"{output_folder}/cost_function.png")

    print("done writing results", flush=True)


def solve_inverse_problem(
    mesh,
    dc_facets,
    ceiling,
    u_data,
    output_folder,
    iteration_pattern,
    constant_controls=False,
):
    TH = def_state_space(mesh)
    U = df.FunctionSpace(mesh, "CG", 1)

    if constant_controls:
        active_strain = [da.Constant(0.0) for _ in range(len(u_data))]
        theta = da.Constant(0.0)
    else:
        active_strain = [da.Function(U) for _ in range(len(u_data))]
        
        for active in active_strain:
            active.assign(da.Constant(0.0))

        theta = da.Function(U)
        theta.assign(da.Constant(0.0))

    states = [da.Function(TH) for _ in range(len(u_data))]

    cost_over_outer = []

    V = df.VectorFunctionSpace(mesh, "CG", 2)
    bcs = def_bcs(dc_facets, TH)
    o = 0

    for iters_inner in iteration_pattern:
        o += 1
        print(f"Starting new inverse problem: {o} / {len(iteration_pattern)}.")

        # now we have a solution; try finding it solving an inverse problem
        (active_strain, theta, states, cost_over_inner,) = solve_inverse_problem_inner(
            mesh,
            ceiling,
            u_data,
            active_strain,
            theta,
            states,
            TH,
            U,
            bcs,
            constant_controls,
            iters_inner,
        )

        if o > 1:
            prev_cost = cost_over_outer[-1]
            cur_cost = cost_over_inner[-1]

        cost_over_outer += cost_over_inner

        if output_folder is not None:
            output_folder_it = output_folder + "/outer_iteration_" + str(o)
            write_results(
                U,
                V,
                u_data,
                active_strain,
                theta,
                states,
                cost_over_outer,
                output_folder_it,
            )

        eps = 1e-6
        if o > 1 and abs(prev_cost - cur_cost) < eps:
            print(f"cost function didn't improve (< {eps} in difference)")
            print(f"stopping at outer iteration {o}")
            break

    return active_strain, theta
