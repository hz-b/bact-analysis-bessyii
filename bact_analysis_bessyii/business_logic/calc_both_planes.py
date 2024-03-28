import logging
from typing import Sequence

import numpy as np

from bact_math_utils.stats import mean_square_error, mean_absolute_error
from bact_math_utils.distorted_orbit import closed_orbit_distortion
from ..business_logic.calc import angle
from ..model.analysis_util import get_data_as_lists
from ..model.analysis_model import (
    MagnetEstimatedAngles,
    FitReadyDataPerMagnet,
    DistortedOrbitUsedForKick,
    ValueForElement,
    DistortedOrbitUsedForKickTransversalPlanes,
    TransversePlanesValuesForElement,
    EstimatedAngles,
    MeasuredValues,
    FitResult,
    ErrorEstimates,
    EstimatedAngleForPlane,
)
from ..model.planes import Planes
from ..model.twiss import Twiss

logger = logging.getLogger("bact-analysis-bessyii")


def get_magnet_estimated_angles_both_planes(
    fit_ready_data: FitReadyDataPerMagnet,
    pos_name: str,
    selected_model: Twiss,
    t_theta: float,
    str: str = "pos",
    rms: str = "rms",
) -> MagnetEstimatedAngles:
    """Fit both planes at once: e.g. the exciation of a steerer

    Args:
        fit_ready_data:
        pos_name:
        selected_model:
        t_theta:
        pos:
        str:
        rms:

    Returns:

    Todo:
        Shall one set t_theta to 1?
        Or shall one even drop it
    """
    name = fit_ready_data.name

    dist_orbs = DistortedOrbitUsedForKickTransversalPlanes(
        delta=[
            TransversePlanesValuesForElement(x=x, y=y)
            for x, y in zip(
                *[
                    estimate_orbit_for_plane(
                        selected_model, pos_name, plane, t_theta
                    ).delta
                    for plane in (Planes.x, Planes.y)
                ]
            )
        ],
        kick_strength=t_theta,
    )

    eax, eay = derive_angles_both_planes(
        measured_data_x=fit_ready_data.x,
        measured_data_y=fit_ready_data.y,
        excitations=fit_ready_data.excitations,
        orbits=dist_orbs,
    )
    return MagnetEstimatedAngles(x=eax, y=eay, name=name)


def estimate_orbit_for_plane(
    selected_model: Twiss, pos_name: str, plane: Planes, t_theta: float
) -> DistortedOrbitUsedForKick:
    dist_orb = closed_orbit_distortion(
        selected_model.get_plane(plane).beta,
        selected_model.get_plane(plane).nu * 2 * np.pi,
        tune=selected_model.get_plane(plane).nu[-1],
        beta_i=selected_model.at_position(pos_name).get_plane(plane).beta,
        mu_i=selected_model.at_position(pos_name).get_plane(plane).nu * 2 * np.pi,
        theta_i=t_theta,
    )

    return DistortedOrbitUsedForKick(
        kick_strength=t_theta,
        delta=[
            ValueForElement(val=val, name=name)
            for name, val in zip(selected_model.names, dist_orb)
        ],
    )


def derive_angles_both_planes(
    measured_data_x: Sequence[MeasuredValues],
    measured_data_y: Sequence[MeasuredValues],
    excitations: Sequence[float],
    orbits: DistortedOrbitUsedForKickTransversalPlanes,
) -> (EstimatedAngles, EstimatedAngles):
    """Kicker angle derived from expected orbit, excitation and distortion measurements

    Args:
        orbits:          orbit expected for some excitation (e.g. 10 urad)
        excitations:     different excitations applied to the magnet
        measured_data_x: the measured orbit distortions (containing
                             difference orbit), x_plane
        measured_data_y: the measured orbit distortions (containing
                             difference orbit), y_plane

    Returns:
        angle and orbit offsets (value and errors)

    Warning:
        code currently assumes that there are the same number of
        points in x and y
    """
    measurement_x, weights_x = np.array(get_data_as_lists(measured_data_x))
    measurement_y, weights_y = np.array(get_data_as_lists(measured_data_y))
    measurements = np.hstack([measurement_x, measurement_y])
    weights = np.hstack([weights_x, weights_y])
    excitations = np.asarray(excitations)

    # plane x and y do not necessarily consist of the same set of point
    orbit_x_at_measurement_points = np.asarray(
        [orbits.at_position(datum.name).x.val for datum in measured_data_x[0].data]
    )
    orbit_y_at_measurement_points = np.asarray(
        [orbits.at_position(datum.name).y.val for datum in measured_data_y[0].data]
    )
    orbit_xy_at_measurement_points = np.hstack(
        [orbit_x_at_measurement_points, orbit_y_at_measurement_points]
    )
    if weights is None:
        sqw = None
    else:
        sqw = np.sqrt(weights)
    #: todo: reenable weights
    sqw = None

    # check that these both are vectors
    (n_exc,) = excitations.shape
    (n_orb_x,) = orbit_x_at_measurement_points.shape
    (n_orb_y,) = orbit_y_at_measurement_points.shape

    n_orb = n_orb_x + n_orb_y

    def prep_A(n_exc, n_orb):
        # now lets do it 2 times for x and y

        # prepare the left hand side of the fit ... 2 steps
        # step 1
        # dimensions: (x,y), step, pos, parameter
        # x and y is put first so that the whole code can be as similar to
        # the 1D code as possible
        A_prep = np.zeros([n_exc, n_orb, n_orb + 1], dtype=np.float_)
        # mark the data appropriately to fit the beam position monitor offsets
        idx = np.arange(n_orb)
        # for the bpm fit ....excitations[np.newaxis, :, np.newaxis] * orbit_xy_at_measurement_points[:, np.newaxis, :]
        # fit angle is put at the last position
        A_prep[:, idx, idx] = 1.0

        # step 2: put the scaled orbit in place
        # Scale independents matrix with weights
        sorb = (
            excitations[:, np.newaxis] * orbit_xy_at_measurement_points[np.newaxis, :]
        )
        if sqw is not None:
            sorb = sorb * sqw
        A_prep[:, :, -1] = sorb
        return A_prep

    A_prep = prep_A(n_exc, n_orb)

    # todo: check that the coordinates are in the appropriate order
    # orbit should be now in only one dimension ... fitting x and y in one stretch
    A = A_prep.reshape(-1, n_orb + 1)

    # scale measured data "b" column
    if sqw is not None:
        try:
            measurements = measurements * sqw
        except Exception as exc:
            txt = (
                f"{__name__}.derive_angle: Failed to apply weights: {exc}"
                f" measurements.dims : {measurements.dims} weight dims {sqw.dims}"
            )
            logger.error(txt)
            raise exc

    b = np.ravel(measurements)
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

    # only one parameter is used for the magnet strength
    equivalent_angle = FitResult(
        value=p[-1] * orbits.kick_strength,
        std=p_std[-1] * orbits.kick_strength,
        name="equivalent_angle",
        #: todo ... need to beef it up to separate it for the x any y part
        # furthermore should contain all input
        # FitInput(A=A_prep, b=b)
        input=None,
    )

    bpm_offsets_x = [
        FitResult(value=v, std=s, name=datum.name, input=None)
        for v, s, datum in zip(p[:n_orb_x], p_std[:n_orb_x], measured_data_x[0].data)
    ]
    # last one is the equivalent angle
    bpm_offsets_y = [
        FitResult(value=v, std=s, name=datum.name, input=None)
        for v, s, datum in zip(
            p[n_orb_x:-1], p_std[n_orb_x:-1], measured_data_y[0].data
        )
    ]

    # calculate estimates for fit: mse / mae
    # first scale the prep matrix using parameters
    dv = A @ p - b
    # split it up to the shape that the measurement had
    # one value per measurement line
    dv = dv.reshape(measurements.shape)
    dv_x, dv_y = dv[:, :n_orb_x], dv[:, n_orb_x:]
    # one per excitation

    estimated_angle_x = EstimatedAngleForPlane(
        orbit=DistortedOrbitUsedForKick(
            delta=[
                ValueForElement(val=val, name=datum.name)
                for val, datum in zip(
                    orbit_x_at_measurement_points, measured_data_x[0].data
                )
            ],
            kick_strength=orbits.kick_strength,
        ),
        equivalent_angle=equivalent_angle,
        bpm_offsets=bpm_offsets_x,
        error_estimates=ErrorEstimates(
            mean_square_error=mean_square_error(dv_x, axis=-1).tolist(),
            mean_absolute_error=mean_absolute_error(dv_x, axis=-1).tolist(),
        ),
    )
    estimated_angle_y = EstimatedAngleForPlane(
        orbit=DistortedOrbitUsedForKick(
            delta=[
                ValueForElement(val=val, name=datum.name)
                for val, datum in zip(
                    orbit_y_at_measurement_points, measured_data_y[0].data
                )
            ],
            kick_strength=orbits.kick_strength,
        ),
        equivalent_angle=equivalent_angle,
        error_estimates=ErrorEstimates(
            mean_square_error=mean_square_error(dv_x, axis=-1).tolist(),
            mean_absolute_error=mean_absolute_error(dv_x, axis=-1).tolist(),
        ),
        bpm_offsets=bpm_offsets_y,
    )

    return estimated_angle_x, estimated_angle_y
