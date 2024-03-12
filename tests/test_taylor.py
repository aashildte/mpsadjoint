import dolfin as df
import dolfin_adjoint as da

from mpsadjoint.cardiac_mechanics import (
    define_state_space,
    define_weak_form,
    define_bcs,
    solve_forward_problem,
    set_fenics_parameters,
)

from mpsadjoint.mesh_setup import Geometry


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


def create_forward_problem(TH, active, theta, geometry, bcs):
    set_fenics_parameters()

    R, state = define_weak_form(TH, active, theta, geometry.ds)

    solve_forward_problem(R, state, bcs)

    u, _ = state.split()
    return u


def errornorm(u, u_synthetic, mesh):
    norm = lambda f: da.assemble(df.inner(f, f) * df.dx(mesh))
    return norm(u - u_synthetic) / norm(da.Constant(1))


def test_taylor_unit_cube():
    TH, bcs, geometry = setup_mesh_funspaces()
    mesh = geometry.mesh

    V, _ = TH.split()
    V = V.collapse()
    u_synthetic = da.Function(V)

    U = df.FunctionSpace(mesh, "CG", 1)

    active_ctrl = da.Function(U)
    active_ctrl.interpolate(da.Constant(0.01))

    theta_ctrl = da.Function(U)
    theta_ctrl.interpolate(da.Constant(0.01))

    h = [da.Function(U), da.Function(U)]
    h[0].interpolate(da.Constant(0.01))
    h[1].interpolate(da.Constant(0.01))

    u = create_forward_problem(TH, active_ctrl, theta_ctrl, geometry, bcs)

    J = errornorm(u, u_synthetic, mesh)

    reduced_functional = da.ReducedFunctional(J, [da.Control(active_ctrl), da.Control(theta_ctrl)])

    results = da.taylor_to_dict(reduced_functional, [da.Function(U), da.Function(U)], h)

    assert min(results["R0"]["Rate"]) > 0.9
    assert min(results["R1"]["Rate"]) > 1.9
    assert min(results["R2"]["Rate"]) > 2.9


if __name__ == "__main__":
    test_taylor_unit_cube()
