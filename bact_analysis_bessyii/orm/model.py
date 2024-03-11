from dataclasses import dataclass
from typing import Sequence
from ..model.analysis_model import FitResult


@dataclass
class FitResultBPMPlane:
    slope: FitResult
    # or intercept ?
    offset: FitResult


@dataclass
class FitResultPerBPM:
    x: FitResultBPMPlane
    y: FitResultBPMPlane
    #: name of the beam position monitor
    name: str


@dataclass
class FitResultPerMagnet:
    data: Sequence[FitResultPerBPM]
    #: name of the magnet that was used to excite the beam
    name: str


@dataclass
class FitResultAllMagnets:
    data: Sequence[FitResultPerMagnet]
