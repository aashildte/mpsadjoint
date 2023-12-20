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


def forward_problem_material_parameters(TH, ds, bcs, a, b, af, bf):
    set_fenics_parameters()

    active = da.Constant(0.05)
    theta = da.Constant(0)

    material_parameters = {"a": a, "b": b, "af": af, "bf": bf}

    R, state = define_weak_form(
        TH,
        active,
        theta,
        ds,
        material_parameters,
    )

    solve_forward_problem(R, state, bcs)

    u, _ = state.split()
    return u


def test_material_parameters():
    TH, bcs, geometry = setup_mesh_funspaces()
    mesh = geometry.mesh
    ds = geometry.ds

    a = da.Constant(2.28)
    af = da.Constant(1.686)
    b = da.Constant(9.726)
    bf = da.Constant(15.779)

    u_standard = forward_problem_material_parameters(TH, ds=ds, bcs=bcs, a=a, b=b, af=af, bf=bf)

    u_a = forward_problem_material_parameters(TH, ds, bcs, a=da.Constant(2 * a), b=b, af=af, bf=bf)
    u_b = forward_problem_material_parameters(TH, ds, bcs, a=a, b=da.Constant(2 * b), af=af, bf=bf)
    u_af = forward_problem_material_parameters(TH, ds, bcs, a=a, b=b, af=da.Constant(2 * af), bf=bf)
    u_bf = forward_problem_material_parameters(TH, ds, bcs, a=a, b=b, af=af, bf=da.Constant(2 * bf))

    norm = lambda f: df.assemble(df.inner(f, f) * df.dx(mesh))
    volume = norm(da.Constant(1))

    diff_a = norm(u_standard - u_a) / volume
    diff_b = norm(u_standard - u_b) / volume
    diff_af = norm(u_standard - u_af) / volume
    diff_bf = norm(u_standard - u_bf) / volume

    # print(diff_a, diff_b, diff_af, diff_bf)

    assert diff_a > 0, "Error: No sensitivity for parameter a."
    assert diff_b > 0, "Error: No sensitivity for parameter b."
    assert diff_af > 0, "Error: No sensitivity for parameter af."
    assert diff_bf > 0, "Error: No sensitivity for parameter bf."


if __name__ == "__main__":
    test_material_parameters()
