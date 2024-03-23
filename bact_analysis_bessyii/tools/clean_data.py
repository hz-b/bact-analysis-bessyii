from typing import Set

import numpy as np

from bact_analysis_bessyii.model.analysis_model import FitReadyData, FitReadyDataPerMagnet


def fit_ready_data_per_magnet_clip_steps(data: FitReadyDataPerMagnet, steps: Set[int]):
    indices = [cnt for cnt, step in enumerate(data.steps) if step in steps]
    return FitReadyDataPerMagnet(
        steps=np.take(data.steps, indices),
        excitations=np.take(data.excitations, indices),
        #: todo: need to handle that data.x can be None
        x=np.take(data.x, indices),
        #: todo: need to handle that data.x can be None
        y=np.take(data.y, indices),
        name=data.name,
        bpm_pos=data.bpm_pos
    )