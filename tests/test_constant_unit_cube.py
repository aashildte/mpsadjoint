import math
import dolfin as df
import dolfin_adjoint as da

from mpsadjoint import (
    set_fenics_parameters,
    define_state_space,
    define_bcs,
    define_weak_form,
    solve_forward_problem,
    solve_inverse_problem,
)

from mpsadjoint.mesh_setup import Geometry

set_fenics_parameters()


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


def test_constant_high():
    # generate synthetic data
    TH, bcs, geometry = setup_mesh_funspaces()

    active = da.Constant(0.03)
    theta = da.Constant(0.0)

    # generate synthetic data

    R, state = define_weak_form(TH, active, theta, geometry.ds)
    solve_forward_problem(R, state, bcs)

    # increase active
    for value in [0.06, 0.08, 0.1, 0.12]:
        active.assign(value)
        solve_forward_problem(R, state, bcs)

    u_synthetic, _ = state.split()

    # then solve the inverse problem

    active_m, _, _ = solve_inverse_problem(
        geometry=geometry,
        u_data=[u_synthetic],
    )

    active_avg = df.assemble(active_m[0] * df.dx) / df.assemble(
        da.Constant(1) * df.dx(geometry.mesh)
    )

    assert math.isclose(
        active_avg, active, abs_tol=0.1
    ), "Error: Could not solve constant problem"


def test_constant():
    TH, bcs, geometry = setup_mesh_funspaces()

    active = da.Constant(0.01)
    theta = da.Constant(0.0)

    # generate synthetic data

    R, state = define_weak_form(TH, active, theta, geometry.ds)

    solve_forward_problem(R, state, bcs)

    u_synthetic, _ = state.split()

    # then consider the inverse problem

    active_m, _, _ = solve_inverse_problem(
        geometry=geometry,
        u_data=[u_synthetic],
    )

    active_avg = df.assemble(active_m[0] * df.dx) / df.assemble(
        da.Constant(1) * df.dx(geometry.mesh)
    )

    assert math.isclose(
        active_avg, active, abs_tol=0.001
    ), "Error: Could not solve constant problem"


if __name__ == "__main__":
    test_constant()
    test_constant_high()
