"""
You need to install the following dependencies

python3 -m pip install opencv-python dask[array,diagnostics] numpy

You also need the `mps` package for reading the data
"""
import cv2
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import mps
import matplotlib.pyplot as plt


def to_uint8(img):
    img_float = img.astype(float)
    return (256 * (img_float / max(img_float.max(), 1e-12))).astype(np.uint8)


def flow(
    image: np.ndarray,
    reference_image: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
):
    """Compute the optical flow using the Farneback method from
    the reference frame to another image

    Parameters
    ----------
    image : np.ndarray
        The target image
    reference_image : np.ndarray
        The reference image
    pyr_scale : float, optional
        parameter, specifying the image scale (<1) to build pyramids
        for each image; pyr_scale=0.5 means a classical pyramid,
        where each next layer is twice smaller than the previous
        one, by default 0.5
    levels : int, optional
        number of pyramid layers including the initial image; levels=1
        means that no extra layers are created and only the original
        images are used, by default 3
    winsize : int, optional
        averaging window size; larger values increase the algorithm
        robustness to image noise and give more chances for fast motion
        detection, but yield more blurred motion field, by default 15
    iterations : int, optional
        number of iterations the algorithm does at each pyramid level, by default 3
    poly_n : int, optional
        size of the pixel neighborhood used to find polynomial expansion in each pixel.
        larger values mean that the image will be approximated with smoother surfaces,
        yielding more robust algorithm and more blurred motion field,
        typically poly_n =5 or 7., by default 5
    poly_sigma : float, optional
        standard deviation of the Gaussian that is used to smooth derivatives used as a
        basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1,
        for poly_n=7, a good value would be poly_sigma=1.5, by default 1.2
    flags : int, optional
         By default 0. operation flags that can be a combination of the following:
         - OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
         - OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter
            instead of a box filter of the same size for optical flow estimation;
            usually, this option gives z more accurate flow than with a box filter,
            at the cost of lower speed; normally, winsize for a Gaussian window should
            be set to a larger value to achieve the same level of robustness.

    Returns
    -------
    np.ndarray
        The motion vectors
    """
    if image.dtype != "uint8":
        image = to_uint8(image)
    if reference_image.dtype != "uint8":
        reference_image = to_uint8(reference_image)

    return cv2.calcOpticalFlowFarneback(
        reference_image,
        image,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )


def get_displacements(
    frames,
    reference_image: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 30,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
):
    """Compute the optical flow using the Farneback method from
    the reference frame to all other frames

    Parameters
    ----------
    frames : np.ndarray
        The frames with some moving objects
    reference_image : np.ndarray
        The reference image
    pyr_scale : float, optional
        parameter, specifying the image scale (<1) to build pyramids
        for each image; pyr_scale=0.5 means a classical pyramid,
        where each next layer is twice smaller than the previous
        one, by default 0.5
    levels : int, optional
        number of pyramid layers including the initial image; levels=1
        means that no extra layers are created and only the original
        images are used, by default 3
    winsize : int, optional
        averaging window size; larger values increase the algorithm
        robustness to image noise and give more chances for fast motion
        detection, but yield more blurred motion field, by default 15
    iterations : int, optional
        number of iterations the algorithm does at each pyramid level, by default 3
    poly_n : int, optional
        size of the pixel neighborhood used to find polynomial expansion in each pixel.
        larger values mean that the image will be approximated with smoother surfaces,
        yielding more robust algorithm and more blurred motion field,
        typically poly_n =5 or 7., by default 5
    poly_sigma : float, optional
        standard deviation of the Gaussian that is used to smooth derivatives used as a
        basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1,
        for poly_n=7, a good value would be poly_sigma=1.5, by default 1.2
    flags : int, optional
         By default 0. operation flags that can be a combination of the following:
         - OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
         - OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize x winsize filter
            instead of a box filter of the same size for optical flow estimation;
            usually, this option gives z more accurate flow than with a box filter,
            at the cost of lower speed; normally, winsize for a Gaussian window should
            be set to a larger value to achieve the same level of robustness.

    Returns
    -------
    Array
        An array of motion vectors relative to the reference image. If shape of
        input frames are (N, M, T) then the shape of the output is (N, M, T, 2).
    """

    all_flows = []
    for im in np.rollaxis(frames, 2):
        all_flows.append(
            dask.delayed(flow)(
                im,
                reference_image,
                pyr_scale,
                levels,
                winsize,
                iterations,
                poly_n,
                poly_sigma,
                flags,
            ),
        )

    with ProgressBar():
        flows = da.stack(*da.compute(all_flows), axis=2)

    return flows


def get_pacing_intervals(data):
    pacing = data.pacing[:]

    if np.max(pacing) == 0.0:
        return [(0, len(pacing) - 1)]  # no pacing

    interval_start = []

    for i in range(len(pacing) - 1):
        if pacing[i] == 0 and pacing[i + 1] > 0:
            interval_start.append(i)

    interval_stop = [i - 1 for i in interval_start[1:]]

    intervals = []
    for start, stop in zip(interval_start, interval_stop):
        intervals.append((start, stop))

    return np.array(intervals)


def process_displacement(path, reference_time_step):
    data = mps.MPS(path)
    file_id = path.split(".")[0]
    intervals = get_pacing_intervals(data)

    um_per_pixel = data.info["um_per_pixel"]

    dt = data.info["dt"]
    ref = data.frames[:, :, reference_time_step - 5 : reference_time_step].mean(2)

    ref_shape = ref.shape[0] * ref.shape[1]
    # print(np.linalg.norm(ref)/ref_shape)

    u = um_per_pixel * get_displacements(data.frames[:, :, :], reference_image=ref)

    X, Y, T, _ = u.shape

    num_beats = 0
    all_beats = []

    for index in range(reference_time_step, T - 200, 200):
        all_beats.append(u[:, :, index : index + 200])
        num_beats += 1

    avg_beat = np.mean(np.array(all_beats), axis=0)
    avg_beat = average_data_in_time(avg_beat)

    c1 = np.linalg.norm(
        avg_beat, axis=3
    )  # norm over (x,y)-koordinat -> X x Y x T-shape
    c2 = np.mean(c1, axis=(0, 1))  # average over (X,Y) -> T-shape

    time = [dt * t for t in range(0, len(c2))]

    plt.plot(time, c2)

    plt.xlabel("Displacement (µm)")
    plt.ylabel("Time (ms)")

    print("Saving displacement trace to " + file_id + ".png")
    # plt.savefig(file_id + "_beat.png", dpi=300)

    # move time forward as first axis
    u1 = da.swapaxes(avg_beat, 1, 2)
    u2 = da.swapaxes(u1, 0, 1)

    # save dispacement here
    np.save(f"{file_id}_displacement_avg_beat_smoothing_factor_5.npy", u2.compute())


if __name__ == "__main__":
    # process_displacement("experiments/AshildData/20211126_bayK_chipB/Control/20211126-GCaMP80HCF20-BayK_Stream_B01_s1_TL-20.tif", 169)
    process_displacement(
        "experiments/AshildData/20211126_bayK_chipB/10nM/20211126-GCaMP80HCF20-BayK_Stream_B01_s1_TL-20.tif",
        124,
    )
    # process_displacement("experiments/AshildData/20211126_bayK_chipB/100nM/20211126-GCaMP80HCF20-BayK_Stream_B01_s1_TL-20.tif", 141)
    # process_displacement("experiments/AshildData/20211126_bayK_chipB/1000nM/20211126-GCaMP80HCF20-BayK_Stream_B01_s1_TL-20.tif", 154)
    # process_displacement("experiments/AshildData/20220105_omecamtiv_chipB/control/20220105-80GCaMP20HCF-omecamtiv_Stream_B01_s1_TL-20-Stream.tif", 60)
    # process_displacement("experiments/AshildData/20220105_omecamtiv_chipB/1 nM/20220105-80GCaMP20HCF-omecamtiv_Stream_B01_s1_TL-20-Stream.tif", 241)
    # process_displacement("experiments/AshildData/20220105_omecamtiv_chipB/10 nM/20220105-80GCaMP20HCF-omecamtiv_Stream_B01_s1_TL-20-Stream.tif", 196)
    # process_displacement("experiments/AshildData/20220105_omecamtiv_chipB/100nM/20220105-80GCaMP20HCF-omecamtiv_Stream_B01_s1_TL-20-Stream.tif", 249)
    # process_displacement("experiments/AshildData/20220105_omecamtiv_chipB/1000nM/20220105-80GCaMP20HCF-omecamtiv_Stream_B01_s1_TL-20-Stream.tif", 152)

    plt.show()
