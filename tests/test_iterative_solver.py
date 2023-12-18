import numpy as np
import dolfin as df
import dolfin_adjoint as da
import pytest

from mpsadjoint import (
    define_state_space,
    define_bcs,
    define_weak_form,
    solve_forward_problem_iteratively,
    set_fenics_parameters,
)

from mpsadjoint.mesh_setup import Geometry
from mpsadjoint.nonlinearproblem import NonlinearProblem


def setup_mesh_funspaces():
    mesh = da.UnitSquareMesh(1, 1)

    pillar_bcs = df.MeshFunction("size_t", mesh, 1, 0)
    pillar_bcs.array()[:] = 0
    pillar_bcs.array()[0] = 1

    ds = df.Measure("ds", domain=mesh, subdomain_data=pillar_bcs)

    geometry = Geometry(mesh, ds)

    TH = define_state_space(geometry.mesh)
    bcs = define_bcs(TH)

    return TH, bcs, geometry


def create_forward_problem_unit_cube(TH, active, theta, geometry, bcs):
    set_fenics_parameters()

    V, _ = TH.split()

    R, state = define_weak_form(TH, active, theta, geometry.ds)

    problem = NonlinearProblem(R, state, bcs)
    solver = da.NewtonSolver()
    solver.solve(problem, state.vector())

    return state, solver, problem


@pytest.mark.parametrize(
    "peak_active, peak_theta",
    [
        (0.01, 0.01),
        (0.02, 0.02),
    ],
)
def test_iterative_solver(peak_active, peak_theta):
    TH, bcs, geometry = setup_mesh_funspaces()

    U = df.FunctionSpace(geometry.mesh, "CG", 1)

    active = da.Function(U)
    active.interpolate(da.Constant(0.005))
    theta = da.Function(U)
    theta.interpolate(da.Constant(0.005))

    state, solver, problem = create_forward_problem_unit_cube(
        TH, active, theta, geometry, bcs
    )

    original_state_values = state.vector()[:]

    # now with that as a base case, try changing each of active and theta by
    # a little, then back again to the original version

    new_active = da.Function(U)
    new_active.interpolate(da.Constant(peak_active))
    new_theta = da.Function(U)
    new_theta.interpolate(da.Constant(peak_theta))

    init_state = da.Function(TH)
    init_state.assign(state)

    solve_forward_problem_iteratively(
        active,
        theta,
        state,
        solver,
        problem,
        new_active,
        new_theta,
    )

    eps = 1e-6
    assert not np.all(np.isclose(state.vector()[:], original_state_values, atol=eps))

    # then back again
    new_active.interpolate(da.Constant(0.005))
    new_theta.interpolate(da.Constant(0.005))

    solve_forward_problem_iteratively(
        active,
        theta,
        state,
        solver,
        problem,
        new_active,
        new_theta,
    )

    eps = 1e-6
    assert np.all(np.isclose(state.vector()[:], original_state_values, atol=eps))


if __name__ == "__main__":
    test_iterative_solver(0.01, 0.01)
    test_iterative_solver(0.02, 0.02)
