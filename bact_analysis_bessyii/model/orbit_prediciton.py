import functools
from dataclasses import dataclass
from typing import Sequence

from .analysis_model import FitResult, index_for_datum_with_name
from .planes import Planes


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
