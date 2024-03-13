import functools
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
from ..model.analysis_model import FitResult
from numpy.typing import ArrayLike


def index_for_datum_with_name(data) -> dict:
    """build an index for a set of data that contain a name"""
    return {datum.name: cnt for cnt, datum in enumerate(data)}


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

    def get(self, bpm_name) -> FitResultPerBPM:
        @functools.lru_cache
        def indices() -> dict:
            return index_for_datum_with_name(self.data)

        return self.data[indices()[bpm_name]]


@dataclass
class FitResultAllMagnets:
    data: Sequence[FitResultPerMagnet]

    def get(self, magnet_name) -> FitResultPerBPM:
        @functools.lru_cache
        def indices() -> dict:
            return index_for_datum_with_name(self.data)

        return self.data[indices()[magnet_name]]


@dataclass
class OrbitResponseMatrixPlane:
    """if matrix of one plane or all data in a single plane"""

    #: a 2D matrix
    slope: ArrayLike
    #: a 2D matrix
    offset: ArrayLike
    steerers: Sequence[str]
    bpms: Sequence[str]


class Plane(Enum):
    x = "x"
    y = "y"


@dataclass
class OrbitResponseMatrices:
    x: OrbitResponseMatrixPlane
    y: OrbitResponseMatrixPlane

    def get(self, plane_name) -> OrbitResponseMatrixPlane:
        plane = Plane(plane_name)
        if plane == Plane.x:
            return self.x
        elif plane == Plane.y:
            return self.y
        else:
            raise AssertionError("how can I end up here?")


@dataclass
class OrbitResponseMatricesPerSteererPlane:
    horizontal_steerers: OrbitResponseMatrices
    vertical_steerers: OrbitResponseMatrices
