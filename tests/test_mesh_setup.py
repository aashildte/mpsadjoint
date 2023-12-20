import dolfin
from pathlib import Path
import pytest
from mpsadjoint import mesh_setup


def test_load_mesh_h5():
    mesh = dolfin.UnitSquareMesh(2, 2)

    pillar_bcs = dolfin.MeshFunction("size_t", mesh, 1, 0)
    pillar_bcs.array()[:] = 1

    filename = Path("test_file.h5")
    if filename.is_file():
        filename.unlink()

    with dolfin.HDF5File(mesh.mpi_comm(), filename.as_posix(), "w") as h5file:
        h5file.write(mesh, "mesh")
        h5file.write(pillar_bcs, "curves")

    geometry = mesh_setup.load_mesh_h5(filename)
    assert (geometry.mesh.coordinates() == mesh.coordinates()).all()

    ds = geometry.ds
    assert dolfin.assemble(dolfin.Constant(1) * ds(1)) > 0.0

    filename.unlink()


def test_load_mesh_h5_raises_FileNotFoundError_when_file_does_not_exist():
    with pytest.raises(FileNotFoundError):
        mesh_setup.load_mesh_h5("no_exist.h5")


def test_load_mesh_h5_raises_RunTimeError_on_wrong_extension():
    f = Path("no_exist.xml")
    f.touch()
    with pytest.raises(RuntimeError):
        mesh_setup.load_mesh_h5(f)
    f.unlink()


if __name__ == "__main__":
    test_load_mesh_h5()
