import numpy as np
import dolfin as df
import dolfin_adjoint as da

from mpsadjoint.cardiac_mechanics import (
    set_fenics_parameters,
    define_state_space,
    define_bcs,
    define_weak_form,
    solve_forward_problem,
)

from mpsadjoint.mesh_setup import Geometry
from mpsadjoint.inverse import (
    solve_pdeconstrained_optimization_problem,
    cost_function,
)


def setup_mesh_funspaces():
    set_fenics_parameters()

    mesh = da.UnitSquareMesh(1, 1)

    pillar_bcs = df.MeshFunction("size_t", mesh, 1, 0)
    pillar_bcs.array()[:] = 0
    pillar_bcs.array()[0] = 1

    ds = df.Measure("ds", domain=mesh, subdomain_data=pillar_bcs)
    geometry = Geometry(mesh, ds)

    TH = define_state_space(geometry.mesh)
    bcs = define_bcs(TH)

    return TH, bcs, geometry


def test_cost_function_cb():
    """

    Tests that eval_cb_checkpoint and cost_function
    values are equal (in the final iteration).

    """

    TH, bcs, geometry = setup_mesh_funspaces()

    active = da.Constant(0.01)
    theta = da.Constant(0.0)

    R, state = define_weak_form(TH, active, theta, geometry.ds)
    solve_forward_problem(R, state, bcs)

    u_synthetic, _ = state.split()

    # initial guesses

    init_active = da.Constant(0.0)
    init_theta = da.Constant(0.0)
    init_state = da.Function(TH)
    num_iterations = 3

    # then consider the inverse problem

    (
        active_opt,
        theta_opt,
        states,
        tracked_quantities,
    ) = solve_pdeconstrained_optimization_problem(
        geometry,
        [u_synthetic],
        [init_active],
        init_theta,
        [init_state],
        num_iterations,
    )

    u_opt = states[0].split()[0]

    cost_fun_value = cost_function([u_opt], [u_synthetic])
    assert np.isclose(
        cost_fun_value, tracked_quantities[0][-1], atol=1e-6
    ), "Error: Mismatch between calculated cost function and final tracked cost function value"


if __name__ == "__main__":
    test_cost_function_cb()
