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
    fit_ready_data,
    selected_model,
    t_theta,
    pos="pos",
    rms="rms",
    **kwargs,
) -> MagnetEstimatedAngles:
    name = fit_ready_data.name
    return MagnetEstimatedAngles(
        name=name,
        x=get_estimated_angle_for_plane(
            fit_ready_data,
            selected_model,
            plane="x",
            theta=t_theta,
            **kwargs,
        ),
        y=get_estimated_angle_for_plane(
            fit_ready_data,
            selected_model,
            plane="y",
            theta=t_theta,
            **kwargs,
        ),
    )


def get_estimated_angle_for_plane(
    fit_ready_data, selected_model, *, plane, theta, fit_offsets: bool = False
) -> EstimatedAngleForPlane:
    """Function to get estimated angle for a specific plane per magnet"""

    magnet_name = fit_ready_data.name

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

    meas_data = getattr(fit_ready_data, plane)
    excitations = fit_ready_data.excitations
    mag_name = fit_ready_data.name
    if fit_offsets:
        logger.warning("Fitting with offsets")
        return derive_angle(kick, meas_data, excitations, plane, mag_name)
    else:
        logger.warning("Fitting without offsets")
        return derive_angle_subtracting_orbit_at_zero_excitation(
            kick, meas_data, excitations, plane, mag_name
        )


def derive_angle_subtracting_orbit_at_zero_excitation(
    orbit_for_kick: DistortedOrbitUsedForKick,
    measured_data: Sequence[MeasuredValues],
    excitations: np.ndarray,
    plane: str,
    magnet_name: str,
    *,
    minimum_excitation: float = 1e-3,
) -> EstimatedAngleForPlane:

    measurements, weights = np.array(get_data_as_lists(measured_data))
    excitations = np.asarray(excitations)

    idx = np.absolute(excitations) < minimum_excitation

    measurement_position_names = measured_data[0].data.keys()
    ref_measurements = np.mean(measurements[idx], axis=0)
    ref_meas_std = np.std(measurements[idx], axis=0)

    diff_measurements = measurements[~idx] - ref_measurements
    excitations = excitations[~idx]
    r = derive_angle_from_distored_orbit(
        orbit_for_kick,
        diff_measurements,
        weights,
        excitations,
        plane=plane,
        magnet_name=magnet_name,
        measurement_position_names=measurement_position_names,
    )
    r.bpm_offsets = OrderedDictImpl(
        zip(
            measurement_position_names,
            [FitResult(value=v, std=s) for v, s in zip(ref_measurements, ref_meas_std)],
        )
    )
    return r


def derive_angle_from_distored_orbit(
    orbit_for_kick: DistortedOrbitUsedForKick,
    measurement: np.ndarray,
    weights: np.ndarray,
    excitations: np.ndarray,
    *,
    plane,
    magnet_name,
    measurement_position_names: Sequence[str],
) -> EstimatedAngleForPlane:
    """Kicker angle derived from expected orbit, excitation and distortion measurements

    Filter out zero excitations ...

    Args:
        orbit:       orbit expected for some excitation (e.g. 10 urad)
        excitation:  different excitations applied to the magnet
        measurement: the measured orbit distortions (containing
                     difference orbit)
        weights (default =None): weights of the measurements

    Returns:
        angle and orbit offsets (value and errors)
    """

    # todo: consistent naming!
    # Todo extract orbit parameters only for bpms ...
    orbit = np.asarray(
        [orbit_for_kick.delta[name] for name in measurement_position_names]
    )

    if weights is None:
        sqw = None
    else:
        sqw = np.sqrt(weights)
    #: todo: renable weights
    sqw = None

    # check that these both are vectors
    (n_exc,) = excitations.shape
    (n_orb,) = orbit.shape

    # step 2: put the scaled orbit in place
    # Scale independents matrix with weights
    sorb = excitations[:, np.newaxis] * orbit[np.newaxis, :]
    if sqw is not None:
        sorb = sorb * sqw

    # todo: check that the coordinates are in the appropriate order
    A = sorb.reshape(-1, 1)

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

    return EstimatedAngleForPlane(
        orbit=orbit_for_kick,
        equivalent_angle=equivalent_angle,
        bpm_offsets=OrderedDictImpl(
            zip(
                measurement_position_names,
                [FitResult(value=0e0, std=0e0)] * len(measurement_position_names),
            )
        ),
        offset=equivalent_angle_to_offset(equivalent_angle, plane, magnet_name),
        error_estimates=error_estimates,
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
    # step 1: sparse matrix for bpm offset fit
    # dimensions: step, pos, parameter
    A_prep = np.zeros([n_exc, n_orb, n_orb + 1], dtype=np.float_)
    # mark the data appropriately to fit the beam position monitor offsets
    # (fit angle == excitations) are put at the last position
    idx = np.arange(n_orb)
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

    return EstimatedAngleForPlane(
        orbit=orbit_for_kick,
        equivalent_angle=equivalent_angle,
        bpm_offsets=OrderedDictImpl(
            zip(
                measurement_position_names,
                [FitResult(value=v, std=s) for v, s in zip(p[:n_orb], p_std[:n_orb])],
            )
        ),
        offset=equivalent_angle_to_offset(equivalent_angle, plane, magnet_name),
        error_estimates=error_estimates,
    )


def equivalent_angle_to_offset(
    equivalent_angle: FitResult, plane: str, magnet_name: str
):
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
    return offset