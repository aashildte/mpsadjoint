from argparse import ArgumentParser
import numpy as np
import dolfin_adjoint as da
from scipy.special import gamma
from mpsadjoint.mesh_setup import load_mesh_h5
from synthetic_framework import inverse_crime


def parse_cl_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--from_step",
        type=int,
        default=150,
        help="Starting time step (out of 999)",
    )
    parser.add_argument(
        "--to_step",
        type=int,
        default=151,
        help="Final time step (out of 999)",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Stride (consider every _ step)",
    )

    parser.add_argument(
        "--active",
        type=str,
        default="sines",
        help="kind of active strain ('sines' or 'constant') to explore",
    )

    parser.add_argument(
        "--fiber",
        type=str,
        default="sines",
        help="kind of fiber direction ('sines' or 'constant') to explore",
    )

    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0,
        help="Noise level; how much noise we perturbate the synthetic data with",
    )

    parser.add_argument(
        "--noise_level_time",
        type=float,
        default=0.0,
        help="Noise level; how much noise we perturbate the synthetic data with per time step",
    )

    parser.add_argument(
        "--num_iterations_iterative",
        type=int,
        default=100,
        help="How many iterations to use solving the inverse problem, in the iterative phase",
    )

    parser.add_argument(
        "--num_iterations_combined",
        type=int,
        default=100,
        help="How many iterations to use solving the inverse problem, in the combined phase",
    )

    parser.add_argument(
        "--mesh_resolution",
        type=str,
        default="40",
        help="Which mesh to use; given as charactheristic length (like 1p0, 0p5, ...)",
    )

    d = parser.parse_args()

    assert d.active in [
        "sines",
        "constant",
    ], "Error: Unknown kind of active strain distribution."
    assert d.fiber in [
        "sines",
        "constant",
    ], "Error: Unknown kind of fiber direction distribution."
    assert d.from_step < d.to_step, "Error: 'from_step' needs to come before 'to_step'"
    assert 0 <= d.from_step < 999, "Error: 'from_step' out of range."
    assert 0 < d.to_step < 1000, "Error: 'to_step' out of range."
    assert d.noise_level >= 0.0, "Error: Noise level should be a positive number."

    return (
        d.from_step,
        d.to_step,
        d.step_length,
        d.active,
        d.fiber,
        d.noise_level,
        d.noise_level_time,
        d.num_iterations_iterative,
        d.num_iterations_combined,
        d.mesh_resolution,
    )


def active_tension_fun(t):
    theta = 3
    k = 2.5
    ts = 0.03 * t
    baseline_value = 0.0001
    scaling_value = 0.2

    return (
        scaling_value * 1 / (gamma(k) * theta**k) * ts ** (k - 1) * np.exp(-ts / theta)
        + baseline_value
    )


def synthetic_experiment():
    (
        start,
        stop,
        step_length,
        active,
        fiber,
        noise_level,
        noise_level_time,
        num_iterations_iterative,
        num_iterations_combined,
        mesh_res,
    ) = parse_cl_arguments()

    mesh_file = f"meshes4/chip_bayK_clmax_{mesh_res}.h5"
    geometry = load_mesh_h5(mesh_file)

    if fiber == "sines":
        theta = da.Expression("0.5*(sin((x[0] - x[1] + 210)/60))", degree=4)
    else:
        theta = da.Constant(0.5)

    active_over_time = []

    time = np.arange(start, stop, step_length)
    tension_values = active_tension_fun(time)

    for t in tension_values:
        if active == "sines":
            a = da.Expression("t*(1 + sin((x[0] + x[1] + 210)/60))", t=t, degree=4)

        else:
            a = da.Constant(t)

        active_over_time.append(a)

    output_folder = (
        f"new_experiments/inverse_synthetic_{mesh_res}_{start}_{stop}_{step_length}"
        + f"_{active}_{fiber}_{noise_level}_{noise_level_time}"
    )

    inverse_crime(
        geometry,
        active_over_time,
        theta,
        output_folder,
        num_iterations_iterative,
        num_iterations_combined,
        noise_level_outer=noise_level,
        noise_level_inner=noise_level_time,
    )


if __name__ == "__main__":
    synthetic_experiment()
