import functools
from dataclasses import dataclass
from typing import Union, Sequence

from ..model.planes import Planes
from ..model.twiss import Twiss

from bact_analysis_bessyii.model.analysis_model import FitResult, index_for_datum_with_name


@dataclass
class Position:
    value: float
    name: str

@dataclass
class SurveyPositions:
    positions: Sequence[Position]

    def get(self, pos_name: str):
        @functools.lru_cache
        def indices():
            return index_for_datum_with_name(self.positions)

        return self.positions[indices()[pos_name]]


@dataclass
class AcceleratorDescription:
    twiss: Twiss
    survey: SurveyPositions


@dataclass
class OrbitPredictionForPlane:
    orbit: Sequence[FitResult]
    excitation: float


@dataclass
class OrbitPredictionForKicks:
    x: Sequence[OrbitPredictionForPlane]
    y: Sequence[OrbitPredictionForPlane]
    name: str

    def get(self, plane: Planes) -> Sequence[OrbitPredictionForPlane]:
        plane = Planes(plane)
        if plane == Planes.x:
            return self.x
        elif plane == Planes.y:
            return self.y
        else:
            raise AssertionError("Santiy check: should not end up here")


@dataclass
class OrbitPredictionCollection:
    per_magnet : Sequence[OrbitPredictionForKicks]

    def get(self, magnet_name: str):
        @functools.lru_cache
        def indices():
            return index_for_datum_with_name(self.per_magnet)
        return self.per_magnet[indices()[magnet_name]]


@dataclass
class StateSpace:
    x: float
    y: float
    pos_name : Union[str, None]
