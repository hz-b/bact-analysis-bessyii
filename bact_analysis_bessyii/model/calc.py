import logging

from bact_analysis_bessyii.model.analysis_model import MeasuredValues, DistortedOrbitUsedForKick, EstimatedAngleForPlane, FitResult
from typing import Sequence
from collections import OrderedDict as OrderedDictmpl
from scipy.linalg import lstsq
import numpy as np

from bact_analysis_bessyii.model.analysis_util import get_data_as_lists
from bact_math_utils.linear_fit import x_to_cov, cov_to_std

logger = logging.getLogger("bact-analysis")


def angle(dist_orb: np.ndarray, meas_orb: np.ndarray) -> (np.ndarray, np.ndarray):
    """Estimate angle using kick model

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


def derive_angle(
    orbit_for_kick: DistortedOrbitUsedForKick,
    measured_data: Sequence[MeasuredValues],
    excitations: np.ndarray
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
    orbit = np.asarray([orbit_for_kick.delta[name.lower()] for name in measurement_position_names])

    # Todo extract orbit parameters only for bpms ...

    if weights is None:
        sqw = None
    else:
        sqw = np.sqrt(weights)
    #: todo: renable weights
    sqw = None

    # check that these both are vectors
    n_exc, = excitations.shape
    n_orb, = orbit.shape

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
    A_prep[:, :, -1] =  sorb


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


    p, p_std  = result
    # assuming that only one parameter is used
    # todo: alternative use name length
    return EstimatedAngleForPlane(orbit=orbit_for_kick,
                                  equivalent_angle=FitResult(value=p[-1] * orbit_for_kick.kick_strength, std=p_std[-1] * orbit_for_kick.kick_strength),
                                  bpm_offsets=OrderedDictmpl(zip(measurement_position_names, [FitResult(value=v, std=s) for v, s in zip(p[:n_orb], p_std[:n_orb])])),
                                  offset=None
                                  )