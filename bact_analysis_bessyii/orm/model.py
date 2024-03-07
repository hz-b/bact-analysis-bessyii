from dataclasses import dataclass

from typing import Dict

from ..model.analysis_model import FitResult


@dataclass
class FitResultPerBPM:
    slope: FitResult
    # or intercept ?
    offset: FitResult


@dataclass
class FitResultPerMagnet:
    x: Dict[str, FitResultPerBPM]
    y: Dict[str, FitResultPerBPM]


@dataclass
class FitResultAllMagnets:
    data: Dict[str, FitResultPerMagnet]
