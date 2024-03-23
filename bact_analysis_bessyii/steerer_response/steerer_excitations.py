"""if only steerer excitations need to be handled

a bit simple rather to test the model
"""
import numpy as np

from ..interfaces.device_location import DeviceLocationServiceInterface
from ..interfaces.space_mapping import SpaceMappingCollectionInterface
from ..model.analysis_model import (
    FitReadyData,
    FitReadyDataPerMagnet,
    MagnetEstimatedAngles,
    EstimatedAngles,
)
from dt4acc.model.twiss import Twiss

from ..model.calc import get_magnet_estimated_angle


def fit_steerer_response_one(
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


def fit_steerer_response_all(
    measured_data: FitReadyData,
    space_col: SpaceMappingCollectionInterface,
    name_pos_service: DeviceLocationServiceInterface,
    twiss: Twiss,
):
    return EstimatedAngles(
        per_magnet={
            data.name: fit_steerer_response_one(data, name_pos_service.get_location_name(data.name), space_col, twiss)
            for data in measured_data.per_magnet
        },
        md=None
    )
