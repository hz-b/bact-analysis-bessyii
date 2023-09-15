import copy
import logging

from bact_analysis_bessyii.model.analysis_model import (
    MeasuredValues,
    DistortedOrbitUsedForKick,
    EstimatedAngleForPlane,
    FitResult,
)
from typing import Sequence
from scipy.linalg import lstsq
import numpy as np
from collections import OrderedDict as OrderedDictImpl
from bact_analysis_bessyii.model.analysis_util import get_data_as_lists, flatten_for_fit
from bact_math_utils.distorted_orbit import closed_orbit_distortion
from bact_math_utils.linear_fit import x_to_cov, cov_to_std

logger = logging.getLogger("bact-analysis")

def calculate_angle_to_offset(tf: float, length: float, polarity: int, alpha: float, tf_scale : float=1.0) -> float:
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
    return calculate_angle_to_offset(magnet_info.tf, magnet_info.length, magnet_info.polarity, angle)

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

def get_magnet_estimated_angle(measurement_per_magnet, selected_model,t_theta) -> MagnetEstimatedAngles:
    name = measurement_per_magnet.name
    return MagnetEstimatedAngles(
        name = name,
        x = get_estimated_angle_for_plane("x", name, measurement_per_magnet.per_magnet, selected_model,t_theta ),
        y = get_estimated_angle_for_plane("y", name, measurement_per_magnet.per_magnet, selected_model,t_theta )
    )
def calculate_offset(angle, ):

    pass
def get_estimated_angle_for_plane(plane, magnet_name, per_magnet_measurement, selected_model,t_theta) -> EstimatedAngleForPlane:
    """Function to get estimated angle for a specific plane per magnet
    """
    # Calculate distorted orbit based on provided model data
    distorted_orbit = closed_orbit_distortion(
        selected_model.beta.sel(plane=plane).values, selected_model.mu.sel(plane=plane).values * 2 * np.pi,
        tune=selected_model.mu.sel(plane=plane).values[-1],
        beta_i=selected_model.beta.sel(plane=plane, pos=magnet_name).values,
        mu_i=selected_model.mu.sel(plane=plane, pos=magnet_name).values * 2 * np.pi,
        theta_i=t_theta,
    )
    # one magnet one plane
    kick = DistortedOrbitUsedForKick(kick_strength=t_theta, delta=OrderedDictImpl(
        zip(selected_model.coords['pos'].values, distorted_orbit)))
    # Prepare measured data and perform fitting
    flattened = flatten_for_fit(per_magnet_measurement,magnet_name)
    #return an object of EstimatedAngleForPlane
    return derive_angle(kick, getattr(flattened, plane), flattened.excitations, plane, magnet_name)

def derive_angle(
    orbit_for_kick: DistortedOrbitUsedForKick,
    measured_data: Sequence[MeasuredValues],
    excitations: np.ndarray,
    plane,
    magnet_name
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
        [orbit_for_kick.delta[name.lower()] for name in measurement_position_names]
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
    for i in range(n_orb):
        # fit angle is put at the last position
        A_prep[:, i, i] = 1.0

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
    return EstimatedAngleForPlane(
        orbit=orbit_for_kick,
        equivalent_angle=FitResult(
            value=p[-1] * orbit_for_kick.kick_strength,
            std=p_std[-1] * orbit_for_kick.kick_strength,
        ),
        bpm_offsets=OrderedDictmpl(
            zip(
                measurement_position_names,
                [FitResult(value=v, std=s) for v, s in zip(p[:n_orb], p_std[:n_orb])],
            )
        ),
        offset=None,
    )
