from typing import Sequence

import numpy as np

from ..model.orbit_prediciton import (
    OrbitPredictionForPlane,
    OrbitPredictionForKicks
)
from ..model.analysis_model import (
    MagnetEstimatedAngles,
    EstimatedAngleForPlane,
    FitResult
)


def compute_prediction_per_magnet(
    estimated_angle: MagnetEstimatedAngles,
    excitations: Sequence[float],
    pos_names_for_measurement: Sequence[str],
    magnet_name: str,
) -> OrbitPredictionForKicks:

    return OrbitPredictionForKicks(
        x=compute_prediction_per_magnet_per_plane(
            estimated_angle.x, excitations, pos_names_for_measurement
        ),
        y=compute_prediction_per_magnet_per_plane(
            estimated_angle.y, excitations, pos_names_for_measurement
        ),
        name=magnet_name,
    )


def compute_prediction_per_magnet_per_plane(
    estimated_angle: EstimatedAngleForPlane,
    excitations: Sequence[float],
    pos_names_for_measurement: Sequence[str],
) -> Sequence[OrbitPredictionForPlane]:
    return [
        OrbitPredictionForPlane(
            orbit=compute_prediction_per_magnet_per_plane_per_excitation(
                estimated_angle, dI, pos_names_for_measurement
            ),
            excitation=dI,
        )
        for dI in excitations
    ]


def compute_prediction_per_magnet_per_plane_per_excitation(
    estimated_angle: EstimatedAngleForPlane,
    excitation: float,
    pos_names_of_measurement: Sequence[str],
) -> Sequence[FitResult]:

    orbit_at_measured_points = np.array(
        [estimated_angle.orbit.delta[name] for name in pos_names_of_measurement]
    )
    kick_value = (
            estimated_angle.equivalent_angle.value * excitation
    )
    kick_std = (
            estimated_angle.equivalent_angle.std * excitation
    )
    # the kick value would be the one to use if we now use an orbit
    # corresponding to a kick strength of one
    # The orbit that we have is for the kick strength so the
    # kick value has to be relative to this one
    expected_orbit_value = orbit_at_measured_points * kick_value / estimated_angle.orbit.kick_strength
    expected_orbit_std = np.absolute(orbit_at_measured_points * kick_std / estimated_angle.orbit.kick_strength)
    return [
        FitResult(value=pv, std=pstd, name=name, input=None)
        for pv, pstd, name in zip(
            expected_orbit_value, expected_orbit_std, pos_names_of_measurement
        )
    ]
