import copy
import logging

from ..business_logic.obsolete import (
    get_polarity_by_magnet_name,
    get_length_by_magnet_name,
)
from ..model.analysis_model import (
    MeasuredValues,
    DistortedOrbitUsedForKick,
    EstimatedAngleForPlane,
    FitResult,
    MagnetEstimatedAngles,
    MagnetInfo,
    ErrorEstimates,
)
from ..model.analysis_util import get_data_as_lists, flatten_for_fit
from bact_math_utils.distorted_orbit import closed_orbit_distortion
from bact_math_utils.linear_fit import x_to_cov, cov_to_std
from bact_math_utils.stats import mean_square_error, mean_absolute_error
from scipy.linalg import lstsq
import numpy as np
from collections import OrderedDict as OrderedDictImpl
from typing import Sequence

# todo:  delete me
# just here for debugging
import matplotlib.pyplot as plt

logger = logging.getLogger("bact-analysis")


def calculate_angle_to_offset(
        tf: float, length: float, polarity: int, alpha: float, tf_scale: float = 1.0
) -> float:
    r"""Derive offset from measured specific kick angle

    Args:
        tf:       central quadrupole strength K1  per excitation
                  dK/dI
        length:   magnet length
        polarity: polarity of the circuit
        alpha:    angle per unit excitation
        tf_scale: how to scale the transfer function (e.g. used if
                  muxer auxcillary coils has many more windings than
                  main coil)


    A (quadrupole) offset :math:`\Delta x_{quad}` gives an kick
    angle of

    .. math::

        \frac{\Delta \vartheta}{\Delta I} =
                \frac{\Delta K_1}{\Delta I}\, L \,\Delta x_{quad}

    Here the specific kick angle :math:`\alpha` and the specific
    exitation :math:`t_f` are used.

    .. math::
        \alpha = \frac{\Delta \vartheta}{\Delta I} \qquad
         t_f = \frac{\Delta K_1}{\Delta I}

    Thus one obtains

    .. math::
        \Delta x_{quad} = \frac{\alpha}{L \, t_f}


    Note:
         due to conventions:
                tf is positive for x and negative for y
                if focusing or not focusing is now hidden in the poliarty?

    Todo:
        review handling of tf and polarity
    """

    devisor = tf * tf_scale * polarity * length
    offset = alpha / devisor
    return offset


def angle_to_offset(magnet_info: MagnetInfo, angle: np.ndarray) -> np.ndarray:
    """

    Args:
        angle: equivalent kick angle estimated from beam based alignment
    """
    return calculate_angle_to_offset(
        magnet_info.tf, magnet_info.length, magnet_info.polarity, angle
    )


def angle(dist_orb: np.ndarray, meas_orb: np.ndarray) -> (np.ndarray, np.ndarray):
    """Estimate angle using kick model

    Function to calculate angle between distorted orbit and measured data
    Fits the exictation of the kick and the offset of the quadrupoles
    (thus the ideal orbit does not need to be subtracted beforehand)

    Args:
        dist_orb:          expected distorted orbits: the expected
                           deviation for the different orbit distortions
        measured orbits:   the orbit that were measured.

    The dist_orb needs to be calculated for the different distortions.
    Typically a distorted orbit is multipled with the used magnet excitation.

    Todo:
        find an appropriate name to distinquish between value and error
        result a good one?

        Find the right place to force the measured orbit to a proper dtype
    """

    # enforce that measured orbit is a numpy array
    # todo
    fitres = lstsq(dist_orb, meas_orb)
    _, residues, rank, _ = fitres
    N, p = dist_orb.shape

    if rank != p:
        raise AssertionError(f"Fit with {p} parameters returned a rank of {rank}")

    # only works if using numpy arrays
    cov = x_to_cov(dist_orb, fitres[1], N, p)
    std = cov_to_std(cov)

    return fitres[0], std


def get_magnet_estimated_angle(
        fit_ready_data, selected_model, t_theta, pos="pos", rms="rms"
) -> MagnetEstimatedAngles:
    name = fit_ready_data.name
    return MagnetEstimatedAngles(
        name=name,
        x=get_estimated_angle_for_plane(
            fit_ready_data,
            selected_model,
            plane="x",
            theta=t_theta
        ),
        y=get_estimated_angle_for_plane(
            fit_ready_data,
            selected_model,
            plane="y",
            theta=t_theta
        )
    )



def calculate_offset(
        angle,
):
    pass


def get_estimated_angle_for_plane(
        fit_ready_data,
        selected_model,
        *,
        plane,
        theta
) -> EstimatedAngleForPlane:
    magnet_name = fit_ready_data.name
    """Function to get estimated angle for a specific plane per magnet"""
    # Calculate distorted orbit based on provided model data
    distorted_orbit = closed_orbit_distortion(
        selected_model.beta.sel(plane=plane).values,
        selected_model.mu.sel(plane=plane).values * 2 * np.pi,
        tune=selected_model.mu.sel(plane=plane).values[-1],
        beta_i=selected_model.beta.sel(plane=plane, pos=magnet_name).values,
        mu_i=selected_model.mu.sel(plane=plane, pos=magnet_name).values * 2 * np.pi,
        theta_i=theta,
    )
    # one magnet one plane
    kick = DistortedOrbitUsedForKick(
        kick_strength=theta,
        delta=OrderedDictImpl(
            zip(selected_model.coords["pos"].values, distorted_orbit)
        ),
    )
    # Prepare measured data and perform fitting
    # flattened = flatten_for_fit(per_magnet_measurement, magnet_name, pos=pos, rms=rms)
    # return an object of EstimatedAngleForPlane
    return derive_angle(
        kick, getattr(fit_ready_data, plane), fit_ready_data.excitations, plane, fit_ready_data.name
    )


def derive_angle(
        orbit_for_kick: DistortedOrbitUsedForKick,
        measured_data: Sequence[MeasuredValues],
        excitations: np.ndarray,
        plane,
        magnet_name,
) -> EstimatedAngleForPlane:
    """Kicker angle derived from expected orbit, excitation and distortion measurements

    Args:
        orbit:       orbit expected for some excitation (e.g. 10 urad)
        excitation:  different excitations applied to the magnet
        measurement: the measured orbit distortions (containing
                     difference orbit)
        weights (default =None): weights of the measurements

    Returns:
        angle and orbit offsets (value and errors)
    """
    measurement, weights = np.array(get_data_as_lists(measured_data))
    excitations = np.asarray(excitations)
    # todo: consistent naming!
    measurement_position_names = measured_data[0].data.keys()
    orbit = np.asarray(
        [orbit_for_kick.delta[name] for name in measurement_position_names]
    )

    # Todo extract orbit parameters only for bpms ...

    if weights is None:
        sqw = None
    else:
        sqw = np.sqrt(weights)
    #: todo: renable weights
    sqw = None

    # check that these both are vectors
    (n_exc,) = excitations.shape
    (n_orb,) = orbit.shape

    # prepare the left hand side of the fit ... 2 steps
    # step 1
    # dimensions: step, pos, parameter
    A_prep = np.zeros([n_exc, n_orb, n_orb + 1], dtype=np.float_)
    # mark the data appropriately to fit the beam position monitor offsets
    idx = np.arange(n_orb)
    # fit angle is put at the last position
    A_prep[:, idx, idx] = 1.0

    # step 2: put the scaled orbit in place
    # Scale independents matrix with weights
    sorb = excitations[:, np.newaxis] * orbit[np.newaxis, :]
    if sqw is not None:
        sorb = sorb * sqw
    A_prep[:, :, -1] = sorb

    # todo: check that the coordinates are in the appropriate order
    A = A_prep.reshape(-1, n_orb + 1)

    # scale measured data "b" column
    if sqw is not None:
        try:
            measurement = measurement * sqw
        except Exception as exc:
            txt = (
                f"{__name__}.derive_angle: Failed to apply weights: {exc}"
                f" measurement.dims : {measurement.dims} weight dims {sqw.dims}"
            )
            logger.error(txt)
            raise exc

    b = np.ravel(measurement)
    # excecute fit
    logger.debug("# excitations %s points in orbit %s", n_exc, n_orb)
    logger.debug("calculating angle A.shape %s, b.shape %s", A.shape, b.shape)
    logger.debug("calculating angle A.dtype %s, b.dtype %s", A.dtype, b.dtype)

    try:
        result = angle(A, b)
    except ValueError as exc:
        txt = (
            f" {__name__}:derive_angle"
            f" A  with shape {A.shape}"
            f" b with shape {b.shape}"
            f" A_prep  with shape {A_prep.shape}"
        )
        logger.error(txt)
        raise exc

    p, p_std = result
    # assuming that only one parameter is used
    # todo: alternative use name length
    equivalent_angle = FitResult(
        value=p[-1] * orbit_for_kick.kick_strength,
        std=p_std[-1] * orbit_for_kick.kick_strength,
    )

    # calculate estimates for fit: mse / mae
    # first scale the prep matrix using parameters
    dv = A @ p - b
    # split it up to the shape that the measurement had
    # one value per measurement line
    dv = dv.reshape(measurement.shape)
    # one per excitation
    error_estimates = ErrorEstimates(
        mean_square_error=mean_square_error(dv, axis=-1).tolist(),
        mean_absolute_error=mean_absolute_error(dv, axis=-1).tolist(),
    )

    # quadrupoles: by convention +K for horizontal -K for vertical plane
    plane_sign = dict(x=1, y=-1)[plane]
    magnet_info = MagnetInfo(
        length=get_length_by_magnet_name(magnet_name),
        # nearly the same for all ... to be looked up
        tf=0.01,
        polarity=get_polarity_by_magnet_name(magnet_name) * plane_sign,
    )

    magnet_info = copy.copy(magnet_info)
    tmp = angle_to_offset(
        magnet_info, np.array([equivalent_angle.value, equivalent_angle.std])
    )
    offset = FitResult(value=float(tmp[0]), std=float(tmp[1]))
    return EstimatedAngleForPlane(
        orbit=orbit_for_kick,
        equivalent_angle=equivalent_angle,
        bpm_offsets=OrderedDictImpl(
            zip(
                measurement_position_names,
                [FitResult(value=v, std=s) for v, s in zip(p[:n_orb], p_std[:n_orb])],
            )
        ),
        offset=offset,
        error_estimates=error_estimates
    )


def plot_fit_result(*, fit_parameters, measurement, measured_data, excitations, sorb, magnet_name: str, plane: str):
    p = fit_parameters
    # should go to postprocessing
    if magnet_name in [
        "Q3M1D7R", "Q1M2T7R",
        "Q1M2D6R",
        "Q2M2D4R",
        "Q5M2T1R", "Q4M1T6R",
        "Q3M2T4R",
    ]:
        # if False:
        t_pscale = 1e6
        fig, axes = plt.subplots(4, 1, sharex=True)
        ax, ax_orb_diff, ax_diff, ax_off = axes
        # ax, ax_orb_diff = axes
        ax.set_title(f"Quadrupole {magnet_name} plane {plane}")
        # reference orbit as obtained from measurement
        # yes hysteresis taken into account
        mref = np.mean([measurement[1, :], measurement[-1, :]], axis=0)

        bpm_names = measured_data[0].data.keys()
        for idx, tmp in enumerate(
                zip(
                    excitations, sorb,
                    (measurement.T - p[:-1][:, np.newaxis]).T,
                    (measurement.T - mref[np.newaxis, :].T).T,
                )
        ):
            excitation, scaled_orbit, dv0, dv1 = tmp
            del tmp

            if excitation == 0:
                continue
            pscale = t_pscale
            if excitation < 0:
                # pscale *= -1
                pass
            label = f"$\\Delta I=${excitation} meas {idx}"
            # fmt:off
            dv1p = np.concatenate([dv1[-5:], dv1, dv1[:5]])
            indicesp = np.arange(-5, len(dv1) + 5)
            indices = np.arange(len(dv1))
            bpm_names = list(bpm_names)
            bpm_names_p = bpm_names[:-5] + bpm_names + bpm_names[:5]
            line, = ax.plot(indicesp, dv1p * pscale, ".-", linewidth=0.1, label=label + "(-bpm offset)")
            ax.plot(indices, scaled_orbit * p[-1] * pscale, "+", color=line.get_color(), linewidth=0.1,
                    label=label + "(scaled orbit)")
            ax.plot(dv0 * pscale, ".--", color=line.get_color(), linewidth=0.1, label=label + "(-meas ref orb)")

            ax_orb_diff.plot(indices, (dv1 - scaled_orbit * p[-1]) * pscale, "x-", color=line.get_color(),
                             linewidth=0.1, label=label + "(scaled orbit)")

        ax.set_ylabel(r"dev $\Delta x$, $\Delta y$ [$\mu$m]")
        ax.legend()

        ax_off.plot(measurement[0, :].T * pscale, "x-", linewidth=0.2, label="start measurement")
        ax_off.plot(measurement[-1, :].T * pscale, "+--", linewidth=0.2, label="end measurement")
        ax_off.plot(mref * pscale, "+--", linewidth=0.2, label="ref. orb. measured")
        ax_off.plot(p[:-1] * pscale, ".-.", linewidth=0.2, label="bpm offsets (fit)")
        ax_off.set_ylabel(r"orbit (avg) offset x,y [$\mu$m]")
        ax_off.legend()
        # fmt:on
        ax_diff.plot((measurement[0, :].T - p[:-1]) * pscale, "x-", linewidth=0.2, label="start measurement")
        ax_diff.plot((measurement[-1, :].T - p[:-1]) * pscale, "+--", linewidth=0.2, label="end measurement")
        ax_diff.plot((mref - p[:-1]) * pscale, ".-", linewidth=0.2, label="ref orb. measured")
        ax_diff.set_ylabel(r"orbit offset -fit, $\Delta$x, $\Delta$y [$\mu$m]")
        ax_diff.legend()

        ax_last = ax_diff
        ax_last.set_xticks(indices)
        ax_last.set_xticklabels(bpm_names)
        plt.setp(ax_last.get_xticklabels(), "horizontalalignment", "right", "verticalalignment", "top", "rotation", 45)
        # did_plot = True
