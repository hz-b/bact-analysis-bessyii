from dataclasses import dataclass
from typing import Sequence
from ..model.analysis_model import FitResult
from numpy.typing import ArrayLike

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


@dataclass
class OrbitResponseSubmatrix:
    #: todo: add names of steerers and bpos
    slope: ArrayLike
    offset: ArrayLike

@dataclass
class OrbitResponseBPMs:
    x : OrbitResponseSubmatrix
    y : OrbitResponseSubmatrix


@dataclass
class OrbitResponseSteeres:
    x: OrbitResponseBPMs
    y: OrbitResponseBPMs




@dataclass
class OrbitResponseMatrixPlane:
    """A set of steerers and a set of bpm's
    """
    steerers: Sequence[str]
    bpms: Sequence[str]
    matrix: OrbitResponseBPMs


@dataclass
class OrbitResponseMatrices:
    horizontal_steerers: OrbitResponseMatrixPlane
    vertical_steerers: OrbitResponseMatrixPlane

