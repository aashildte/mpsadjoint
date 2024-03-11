from mpsadjoint.cardiac_mechanics import fiber_direction
import pytest
import numpy as np
import dolfin as df


@pytest.mark.parametrize("theta", [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
def calculate_fiber_direction_norm(theta):
    mesh = df.UnitSquareMesh(2, 2)
    f0 = fiber_direction(theta)

    norm = df.assemble(df.dot(f0, f0) * df.dx(mesh))
    assert np.isclose(norm, 1.0)


def calculate_fiber_direction_right_angle():
    mesh = df.UnitSquareMesh(2, 2)
    e2 = df.as_vector([0.0, 1.0])

    for theta in [-np.pi / 2, np.pi / 2]:
        f0 = fiber_direction(theta)

        norm = df.assemble(df.dot(f0, e2) * df.dx(mesh))

        assert np.isclose(norm, 1.0) or np.isclose(norm, -1.0)


if __name__ == "__main__":
    # calculate_fiber_direction(-np.pi/2)
    # calculate_fiber_direction(-np.pi/4)
    # calculate_fiber_direction(0)
    # calculate_fiber_direction(np.pi/4)
    # calculate_fiber_direction(np.pi/2)
    calculate_fiber_direction_right_angle()
