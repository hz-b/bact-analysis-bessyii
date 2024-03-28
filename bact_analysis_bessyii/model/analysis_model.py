import enum
import functools
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence, Optional, Union

from ..model.planes import Planes
from numpy.typing import ArrayLike
import numpy as np

from bact_device_models.devices.bpm_elem import BpmElem


def index_for_datum_with_name(data) -> dict:
    """build an index for a set of data that contain a name"""
    return {datum.name: cnt for cnt, datum in enumerate(data)}



class Polarity(IntEnum):
    positive = 1
    negative = -1


@dataclass
class MagnetInfo:
    """
    """
    #: length of the magnet
    length: float
    #: transfer function i.e. K per A
    #: todo check if it is K or G
    tf: float
    #: sign to apply to the angle
    #: e.g.
    #:    * for focusing quadrupole x = 1, y = -1
    #:    * for defocusing quadrupole x = -1, y = 1
    polarity: Polarity


@dataclass
class OffsetFitResult:
    """Magnet offset obtained by BBA for one plane

    Fit of the deviated orbit to the beam based alignment procedure

    Todo:
        compare to FitResult
        Should a separate dataclass be used for the offset
    """
    # offset of the (quadrupole magnet)
    value: float
    # estimate of its accuracy
    error: float


@dataclass
class OffsetData:
    """Magnet offset in both planes

    The result a None if not available (e.g. failed fit or measurement)
    """
    # or better horizontal ?
    x: Optional[OffsetFitResult]
    # or better vertical ?
    y: Optional[OffsetFitResult]
    # name of the magnet the data has been calculated for
    name: str


@dataclass
class BPMCalibrationPlane:
    """scale and offset for the bpm (both in millimeter)

    Todo:
        check if calc functionality should be singled out to a filter

    """

    #: from bits to -10/10 volt, assuming that bits are signed
    #: bact2 used 10/(2**15)
    bit2val : float
    #: from volt to meter ... 1.0 for most bpm's
    scale : float
    #: offset to correct meter for
    offset : float
    active : bool

    def __init__(self, *, bit2val = 10/(2**15), scale=1e-3, offset=0.0, active=True):
        self.bit2val = bit2val
        self.scale = scale
        self.offset = offset
        self.active = active

    def to_pos(self, bits):
        """
        Todo:
            cross check with bact2 that the arithmetic is consistent
        """
        return (bits * self.bit2val) * self.scale - self.offset

    def to_rms(self, bits):
        """
        Todo:
            check that the scale is positive ?
        """
        return abs( (bits * self.bit2val) * self.scale )


@dataclass
class BPMCalibration:
    x : BPMCalibrationPlane
    y : BPMCalibrationPlane


@dataclass
class MeasurementPoint:
    """Dsta taken for one (quadrupole) magnet for one excitation
    """
    step: int
    excitation: np.ndarray
    bpm: Sequence[BpmElem]
    # todo: add processing of the tune
    # tune: Sequence[TuneModel]


@dataclass
class MeasurementPerMagnet:
    """A set of data required to calculate the offset of this magnet
    """
    # name of the magnet the data has been estimated for
    name: str
    per_magnet: Sequence[MeasurementPoint]


@dataclass
class MeasurementData:
    """Data or one measurement campaign (typically whole machine)

    I think geometers tend to call that an epoc
    """
    measurement: Sequence[MeasurementPerMagnet]


@dataclass
class MeasuredItem:
    """
    Mathematically speaking a random variable described by its first two momenta

    Todo:
        Replace or derive from FitResult
    """
    #: value returned by the measurement, typically close to the first momentum
    value: float
    #: estimate of its error typically similar to the first momentum
    rms: float
    #: typically the position name
    name: str


@dataclass
class MeasuredValues:
    """Orbit data along the ring, e.g. as measured by the beam position monitors
    """
    data: Sequence[MeasuredItem]

    def get(self, name) -> MeasuredItem:
        @functools.lru_cache()
        def indices():
            return index_for_datum_with_name(self.data)
        return self.data[indices()[name]]


@dataclass
class FitReadyDataPerMagnet:
    """Measured data prepared for the fitting routine
    Todo:
        review if required?

        bpm names as meta data
        bpm_pos too?
    """
    # name of the magnet the data has been estimated for
    name: str
    # sequence number of the measurement (0, 1, 2, 3 ...)
    steps: Sequence[int]
    # excitation that was applied to the magnet (typically
    # the current of the muxer power converter)
    excitations: Sequence[float]
    x: Optional[Sequence[MeasuredValues]]
    y: Optional[Sequence[MeasuredValues]]

    # I don't recall if these are the names of the beam position monitors
    # or their position. If their position these are used for plotting
    # todo: move to meta data
    bpm_pos: np.ndarray
    # don't recall what it is I guess was used as index
    # towards delta or rms
    # quality: str

    def get(self, plane: Planes):
        plane = Planes(plane)
        if plane == Planes.x:
            return self.x
        elif plane == Planes.y:
            return self.y
        else:
            raise AssertionError("Sanity Error, should not end here")


@dataclass(frozen=True)
class FitReadyData:
    per_magnet: Sequence[FitReadyDataPerMagnet]

    def get(self, magnet_name: str) -> FitReadyDataPerMagnet:
        @functools.lru_cache
        def name_to_index():
            return index_for_datum_with_name(self.per_magnet)

        return self.per_magnet[name_to_index()[magnet_name]]


@dataclass
class ValueForElement:
    """
    """
    val: float
    # element_name
    name: str


@dataclass
class DistortedOrbitUsedForKick:
    """Orbit that an equivalent kicker would create

    Todo: include magnet name?
    """

    # The kick strength that was used to produce the kick
    kick_strength: float
    # offsets produced around the orbit by the kick
    # index : name of the element
    # value: orbit deviation at this spot
    # delta: np.ndarray
    # or should it be a hashable i.e a key value
    delta: Sequence[ValueForElement]


@dataclass
class TransversePlanesValuesForElement:
    """
    """
    x: ValueForElement
    y: ValueForElement


@dataclass
class DistortedOrbitUsedForKickTransversalPlanes:
    """Represent
    """
    #: The kick strength that was used to produce the kick
    kick_strength: float
    delta: Sequence[TransversePlanesValuesForElement]

    def at_position(self, name) -> TransversePlanesValuesForElement:
        @functools.lru_cache
        def indices():
            return index_for_datum_with_name([datum.x for datum in self.delta])

        r = self.delta[indices()[name]]
        assert r.x.name == r.y.name
        return r


@dataclass
class FitInput:
    A: ArrayLike
    b: ArrayLike


@dataclass
class FitResult:
    value: float
    std: float
    name: str
    #: these data are rather large, should we always store them?
    input: Optional[FitInput]


class ErrorType(enum.Enum):
    #: mean square error
    mse="mse"
    #: mean absolute error
    mae="mae"


@dataclass
class ErrorEstimates:
    """estimates of fit quality

    Todo:
        use full name or the abbreviations used in stats?
        provide a value for each excitation?
    """
    mean_square_error: Sequence[float]
    mean_absolute_error: Sequence[float]

    def get(self, error_type: ErrorType) -> Sequence[float]:
        error_type = ErrorType(error_type)
        if error_type == ErrorType.mse:
            return self.mean_square_error
        elif error_type == ErrorType.mae:
            return self.mean_absolute_error
        else:
            raise AssertionError(f"don't know error Type: {error_type}")


@dataclass
class EstimatedAngleForPlane:
    orbit: DistortedOrbitUsedForKick
    # the angle that is corresponding to this kick
    equivalent_angle: FitResult
    # retrieved from the fit measurement
    bpm_offsets: Sequence[FitResult]
    error_estimates : ErrorEstimates

@dataclass
class EstimatedAngleAndOffsetsForPlane(EstimatedAngleForPlane):
    # or derived offset ... no need to separate it
    offset: FitResult

    # fit_ready_data : FitReadyData

@dataclass
class MagnetEstimatedAngles:
    #: name of the magnet the angle estimate was made for
    name: str
    x: EstimatedAngleForPlane
    y: EstimatedAngleForPlane

    def get(self, plane: Planes) -> EstimatedAngleForPlane:
        plane = Planes(plane)
        if plane == Planes.x:
            return self.x
        elif plane == Planes.y:
            return self.y
        else:
            raise AssertionError("should not end up here!")


@dataclass
class EstimatedAngles:
    per_magnet: Sequence[MagnetEstimatedAngles]
    # metadata
    md: Optional[object]

    def get(self, magnet_name: str):
        @functools.lru_cache
        def indices():
            return index_for_datum_with_name(self.per_magnet)

        return self.per_magnet[indices()[magnet_name]]


