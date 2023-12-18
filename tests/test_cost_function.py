import math
import dolfin_adjoint as da
import dolfin as df
import pytest
import mpsadjoint

from mpsadjoint.mesh_setup import Geometry


@pytest.mark.parametrize(
    "u_model_arr, expected_cost_diff",
    [
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 1.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 1.0),
    ],
)
def test_cost_diff_constant_u(u_model_arr, expected_cost_diff):
    mesh = da.UnitSquareMesh(3, 3)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    u_model = da.Function(V)
    u_data = da.Function(V)

    exp = da.Expression(("ax", "ay"), ax=u_model_arr[0], ay=u_model_arr[1], element=V.ufl_element())
    u_data.interpolate(exp)
    u_model.assign(da.Constant((0.0, 0.0)))

    geometry = Geometry(
        mesh=mesh,
        ds=None,
    )

    cost_diff = mpsadjoint.inverse.cost_function(geometry, [u_model], [u_data])

    print(cost_diff)

    assert math.isclose(cost_diff, expected_cost_diff)


@pytest.mark.parametrize(
    "u_model_arr, expected_cost_diff",
    [
        ((0.0, 0.0), 2.0),
        ((0.0, 1.0), 2.0),
        ((1.0, 0.0), 2.0),
        ((1.0, 1.0), 2.0),
    ],
)
def test_cost_diff_varying_u(u_model_arr, expected_cost_diff):
    mesh = da.UnitSquareMesh(3, 3)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    u_model = da.Function(V)
    u_data = da.Function(V)

    exp = da.Expression(
        ("ax + x[0]", "ay + x[1]"),
        ax=u_model_arr[0],
        ay=u_model_arr[1],
        element=V.ufl_element(),
    )
    u_data.interpolate(exp)
    u_model.assign(da.Constant((0.0, 0.0)))

    geometry = Geometry(
        mesh=mesh,
        ds=None,
    )

    cost_diff = mpsadjoint.inverse.cost_function(geometry, [u_model], [u_data])

    print(cost_diff)

    assert math.isclose(cost_diff, expected_cost_diff)


def test_cost_function():
    mesh = da.UnitSquareMesh(3, 3)
    # U = df.FunctionSpace(mesh, "CG", 1)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    u_model = da.Function(V)
    u_data = da.Function(V)
    # controls = [da.Function(U)]

    geometry = Geometry(
        mesh=mesh,
        ds=None,
    )
    cost = mpsadjoint.inverse.cost_function(geometry, [u_model], [u_data])

    assert math.isclose(cost, 0.0)


if __name__ == "__main__":
    test_cost_function()
    test_cost_diff_constant_u((0.0, 0.0), 0.0)
    test_cost_diff_constant_u((0.0, 1.0), 1.0)
    test_cost_diff_constant_u((1.0, 0.0), 1.0)
    test_cost_diff_constant_u((1.0, 1.0), 1.0)
    test_cost_diff_varying_u((0.0, 0.0), 2.0)
    test_cost_diff_varying_u((0.0, 1.0), 2.0)
    test_cost_diff_varying_u((1.0, 0.0), 2.0)
    test_cost_diff_varying_u((1.0, 1.0), 2.0)
