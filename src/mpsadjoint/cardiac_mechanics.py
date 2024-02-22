"""

Åshild Telle / Simula Research Laboratory / 2024

"""

import numpy as np

# import ufl (new/old version of fenics)
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

import dolfin as df
import dolfin_adjoint as da


def set_fenics_parameters():
    """

    Sets some fenics parameters – some of these might be redundant if
    you have the latest version of Fenics.

    df.set_log_level gives verbatim output and might be worth
    commenting out for debugging purposes.

    """

    df.set_log_level(60)

    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.parameters["form_compiler"]["representation"] = "uflacs"
    df.parameters["form_compiler"]["quadrature_degree"] = 4


def cond(a, k: float = 100):
    r"""
    Continuous approximation of the conditional term

    .. math::
        a \text{, if } a > 0
        0 \text{otherwise}

    given by a sigmoid function.

    """

    return a / (1 + df.exp(-k * a))


def psi_holzapfel(
    IIFx,
    I4e1,
    a=da.Constant(2.28),
    af=da.Constant(9.726),
    b=da.Constant(1.685),
    bf=da.Constant(15.779),
):
    """
    Transversely isotropic material model, first two terms of the
    material model defined by Holzapfel and Odgen. Material
    parameters taken from Table 1 (third row, biaxial tests)
    in their paper.

    Args:
        IIFx: first invariant, tr(C)
        I4e1: invariant for the fiber direction component
        a : Material parameter, by default da.Constant(2.28)
        af : Material parameter, by default da.Constant(9.726)
        b : Material parameter, by default da.Constant(1.685)
        bf : Material parameter, by default da.Constant(15.779)

    Returns:
        an hyperelastic anisotropic strain energy function

    """

    W_hat = a / (2 * b) * (df.exp(b * (IIFx - 2)) - 1)
    W_f = af / (2 * bf) * (df.exp(bf * cond(I4e1 - 1) ** 2) - 1)

    return W_hat + W_f


def psi_neohookean(IIFx, C10=da.Constant(2000)):
    """NeoHookean strain energy function.

    Args:
        IIFx: first invariant, tr(C)

    Returns:
        neohookean srain energy function

    """

    return C10 / 2 * (IIFx - 2)


def fiber_direction(theta: da.Function):
    """

    Defines fiber direction as a vector space, defined from a given angle.

    Args:
        theta: scalar-valued function, spatial distribution of the angle

    Returns:
        a vector field defined  by rotating [1, 0, 0] according to theta

    """

    R = df.as_matrix(
        ((df.cos(theta), -df.sin(theta)), (df.sin(theta), df.cos(theta)))
    )

    return R * df.as_vector([1.0, 0.0])


def PK_stress_tensor(F, p, active_function, mat_params_tissue, theta):
    """

    Defines the first Piola-Kirchhoff stress tensor, using an active
    strain approach and holzapfel/odgen strain energy function.

    Args:
        F: deformation tensor
        p: hydrostatic pressure
        active_function: active tension (gamma)
        mat_params_tissue: material paraameters
        theta: fiber direction angle

    Returns:
        the first Piola-Kirchhoff stress tensor

    """
    f0 = fiber_direction(theta)
    C = F.T * F
    J = df.det(F)
    Jm1 = pow(J, -1.0)

    I1 = Jm1 * df.tr(C)
    f0 = fiber_direction(theta)
    I4f = Jm1 * df.inner(C * f0, f0)

    mgamma = 1 - active_function
    I1e = mgamma * I1 + (1 / mgamma ** 2 - mgamma) * I4f
    I4fe = 1 / mgamma ** 2 * I4f

    psi = psi_holzapfel(I1e, I4fe, **mat_params_tissue)
    PK1 = df.diff(psi, F) + p * Jm1 * J * df.inv(F.T)

    return PK1


def define_state_space(mesh: da.Mesh):
    """

    Defines a function space over Taylor-Hood (P2-P1) elements, which
    will be used for all mechanical simulations.

    Args:
        Mesh

    Returns:
        Taylor-Hood function space

    """

    P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

    TH = df.FunctionSpace(mesh, df.MixedElement([P2, P1]))

    return TH


def define_bcs(TH: df.FunctionSpace) -> list[da.DirichletBC]:
    """

    Defines Dirichlet boundary conditions. This is sort of a hack;
    Dolfin adjoint doesn't allow no bcs so we are adding an empty set

    Args:
        TH: function space in which the state is defined over; P2-P1

    Returns:
        list of DirichletBC

    """
    V, _ = TH.split()

    xmin_bnd = "0"
    bcs = [da.DirichletBC(V, da.Constant(np.zeros(2)), xmin_bnd)]

    return bcs


def remove_rigid_motion_term(
    mesh: da.Mesh,
    u: da.Function,
    r: da.Function,
    state: da.Function,
    test_state: df.TestFunction,
):
    """

    Defines the part of the weak form which removes rigid motion,
    i.e. non-uniqueness in translation and rotation.

    Args:
        u (dolfin funciton): displacement function
        r (dolfin function): test function for the space of rigid motion
        state, triplet: all three trial functions
        test_state, triplet: all three test functions

    Returns:
        weak form term

    """

    RM = [
        da.Constant((1, 0)),
        da.Constant((0, 1)),
        da.Expression(("-x[1]", "x[0]"), degree=1),
    ]

    Pi = sum(df.dot(u, zi) * r[i] * df.dx for i, zi in enumerate(RM))

    return df.derivative(Pi, state, test_state)


def define_weak_form(
    TH: df.FunctionSpace,
    active_function: list[da.Function],
    theta: da.Function,
    ds: df.MeshFunction,
    mat_params_tissue: dict[str, float] = {},
) -> ufl.form.Form:
    """

    Defines trial and test functions + weak form.

    Args:
        TH: a taylor-hood function space (P2-P1)
        active_function: active tension (gamma)
        theta: fiber direciton angle
        xi_tissue: char. function defining the tissue
        xi_pillars: char. function defining the pillars

    Returns
        weak form: solving for this gives a mechanical equilibrium solution
            state (i.e. displacement, pressure)

    """

    state = da.Function(TH)
    test_state = df.TestFunction(TH)
    u, p = df.split(state)
    v, q = df.split(test_state)

    F = df.variable(df.Identity(2) + df.grad(u))
    J = df.det(F)

    PK1 = PK_stress_tensor(F, p, active_function, mat_params_tissue, theta)

    elasticity_term = df.inner(PK1, df.grad(v)) * df.dx
    pressure_term = q * (J - 1) * df.dx

    robin_value = da.Constant(1.0)
    robin_marker = 1
    robin_bcs_term = df.inner(robin_value * u, v) * ds(robin_marker)

    weak_form = elasticity_term + pressure_term + robin_bcs_term

    return weak_form, state


def solve_forward_problem(R, state, bcs: list[da.DirichletBC]):
    """

    Solves the forward mechanical problem.

    """
    da.solve(
        R == 0,
        state,
        bcs,
        solver_parameters={"newton_solver": {"maximum_iterations": 20}},
    )


def get_control_values(control):
    """

    Subtracts all function values as a vector.

    """
    return control.vector()[:]


def assign_new_values(control_function, new_values):
    """

    Assigns values to function from a vector.

    """
    control_function.vector()[:] = new_values


def solve_forward_problem_iteratively(
    active_function,
    theta_function,
    state_function,
    newtonsolver,
    problem,
    active: da.Function,
    theta: da.Function,
    step_length=0.1,
):
    """
    Solves the system iteratively, going from zero to the one
    given by active and theta (usually the ones found in the
    optimization process). This works fine with scaled as well
    as unscaled parameters.

    In case the system doesn't converge, we'll try again with
    a smaller step length in a recursive manner; if the step
    length reaches an unexpected low value, we'll stop the recursion.

    Args:
        active_function: da.Constant or da.Function instance,
            the active strain for one time step
        theta_function: da.Constant or da.Function instance,
            determining the fiber direction
        state_function: state we solve for; dependent on `active_function`
            and theta_function
        newtonsolver: solver for the corresponding forward problem
        problem: the forward problem defined; which again depends on
            active and theta_function, solution to be saved in state
        active (values): optimal (or changed) values; active_function
            will gradually be changed to these
        theta (values): optimal (or changed) values; theta_function
            will gradually be changed to these
        step_length: float between 0 and 1, how large steps we want
            to use for stepping up to expected values
    """

    assert step_length > 1e-14, "Error: Really low step length – aborting"
    assert step_length <= 1.0, "Error: Step length can't be > 1."

    original_active_values = get_control_values(active_function)
    active_values = get_control_values(active)

    original_theta_values = get_control_values(theta_function)
    theta_values = get_control_values(theta)

    original_state_values = state_function.vector()[:]

    N = int(1 / step_length)

    print(f"Solving system iteratively using {N + 1} steps.")

    try:
        for n in range(N):
            print(f"* Solving for step {n + 1} / {N + 1}")
            new_active_values = (
                original_active_values
                + n
                * step_length
                * (active_values - original_active_values)
            )
            new_theta_values = original_theta_values + n * step_length * (
                theta_values - original_theta_values
            )

            assign_new_values(theta_function, new_theta_values)
            assign_new_values(active_function, new_active_values)
            newtonsolver.solve(problem, state_function.vector())

        # then one final solve
        print(f"* Solving for final step {N + 1} / {N + 1}")

        assign_new_values(theta_function, theta_values)
        assign_new_values(active_function, active_values)
        newtonsolver.solve(problem, state_function.vector())

        print("Done solving!")

    except RuntimeError:
        assign_new_values(active_function, original_active_values)
        assign_new_values(theta_function, original_theta_values)
        assign_new_values(state_function, original_state_values)

        print("Error in iterative solve; trying with a smaller value.")
        print("Step length: ", step_length / 2)
        solve_forward_problem_iteratively(
            active_function,
            theta_function,
            state_function,
            newtonsolver,
            problem,
            active,
            theta,
            step_length / 2,
        )
