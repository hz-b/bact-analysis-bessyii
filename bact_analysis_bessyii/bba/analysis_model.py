from dataclasses import dataclass
import numpy as np
from typing import List, Sequence, OrderedDict

from bact_bessyii_ophyd.devices.pp.bpmElem import BpmElem


# corresponding data array to offset data
# res = xr.DataArray(
#    data=[[x, x_err], [y, y_err]],
#    dims=["plane", "result", "name"],
#    coords=[["x", "y"], ["value", "error"], names],
# )


@dataclass
class OffsetFitResult:
    """Magnet offset obtained by BBA for one plane

    Fit of the deviated orbit to the beam based alignment procedure
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
    x: OffsetFitResult | None
    # or better vertical ?
    y: OffsetFitResult | None
    # name of the magnet the data has been calculated for
    name: str


@dataclass
class PreprocessedDataItem:
    """Dsta taken for one (quadrupole) magnet for one excitation
    """
    step: int
    excitation: np.ndarray
    bpm: Sequence[BpmElem]
    tune: None

@dataclass
class PreprocessedData:
    """Data layout
    """
    # name of the magnet the data has been estimated for
    name: str
    data : Sequence[PreprocessedDataItem]

@dataclass
class FitReadyOrbit:
    """data ready for the fit

    Fit can only deal with 1D arrays
    """

    # the offset (measured using the bpm's)
    delta: np.ndarray
    # estimate of accuracy
    rms: np.ndarray

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
    step: np.ndarray
    # excitation that was applied to the magnet (typically
    # the current of the muxer power converter)
    excitation: np.ndarray
    x: FitReadyOrbit | None
    y: FitReadyOrbit | None
    # I don't recall if these are the names of the beam position monitors
    # or their position. If their position these are used for plotting
    bpm_pos: np.ndarray
    # don't recall what it is I guess was used as index
    # towards delta or rms
    # quality: str

@dataclass
class FitReadyData:
    data : Sequence[FitReadyDataPerMagnet]

@dataclass
class ValueForElement:
    """
    """
    val : float
    # element_name
    name : str

@dataclass
class DistortedOrbitUsedForKick:
    """Orbit that an equivalent kicker would create"""

    # The kick strength that was used to produce the kick
    kick_strength: float
    # offsets produced around the orbit by the kick
    # delta: np.ndarray
    # or should it be a hashable i.e a key value
    delta : OrderedDict
    # delta : Sequence[ValueForElement]

@dataclass
class FitResult:
    value: float
    std: float


@dataclass
class EstimatedAngleForPlane:
    orbit: DistortedOrbitUsedForKick
    # the angle that is corresponding to this kick
    equivalent_angle: FitResult
    # retrieved from the fit measurement
    bpm_offsets: List[FitResult]
    # or derived offset ... no need to separate it
    offset: FitResult


@dataclass
class EstimatedAngle:
    name: str
    x: EstimatedAngleForPlane
    y: EstimatedAngleForPlane