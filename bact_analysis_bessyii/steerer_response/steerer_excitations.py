"""if only steerer excitations need to be handled

a bit simple rather to test the model
"""
import numpy as np


from ..interfaces.space_mapping import SpaceMappingCollectionInterface
from ..model.analysis_model import (
    FitReadyDataPerMagnet,
    MagnetEstimatedAngles
)
from ..model.twiss import Twiss

from bact_analysis_bessyii.business_logic.calc import get_magnet_estimated_angle
from bact_analysis_bessyii.business_logic.calc_both_planes import get_magnet_estimated_angles_both_planes


def fit_steerer_response_one_separate_per_plane(
    measured_data: FitReadyDataPerMagnet,
    pos_name: str,
    space_col: SpaceMappingCollectionInterface,
    model: Twiss,
) -> MagnetEstimatedAngles:
    mapping = space_col.get(measured_data.name)
    k_values = np.array([mapping.inverse(v) for v in measured_data.excitations])
    # todo: ensure that bpm names of data in model
    bpm_names = [datum.name for datum in measured_data.y[0].data]

    return get_magnet_estimated_angle(measured_data, pos_name, model, t_theta=1e-5)


def fit_steerer_response_one_both_planes(
    measured_data: FitReadyDataPerMagnet,
    pos_name: str,
    space_col: SpaceMappingCollectionInterface,
    model: Twiss,
) -> MagnetEstimatedAngles:
    mapping = space_col.get(measured_data.name)
    k_values = np.array([mapping.inverse(v) for v in measured_data.excitations])
    # todo: ensure that bpm names of data in model
    bpm_names = [datum.name for datum in measured_data.y[0].data]

    return get_magnet_estimated_angles_both_planes(measured_data, pos_name, model, t_theta=1e-5)



