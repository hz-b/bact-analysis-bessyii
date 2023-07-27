from dataclasses import dataclass
import numpy as np


@dataclass
class PlaneData:
    x: np.ndarray
    y: np.ndarray


@dataclass
class ResultData:
    value: np.ndarray
    error: np.ndarray


@dataclass
class OffsetData:
    name: str
    plane: PlaneData
    offset_values: ResultData


@dataclass
class PreprocessedData:
    name: str
    step: np.ndarray
    bpm_pos: np.ndarray
    plane: str
    quality: str
    x_pos: np.ndarray
    y_pos: np.ndarray
    x_rms: np.ndarray
    y_rms: np.ndarray
    excitation: np.ndarray


@dataclass
class EstimatedAngle:
    name: str
    plane: np.ndarray
    pos: np.ndarray
    result: np.ndarray
    parameter: np.ndarray
    orbit: np.ndarray
    fit_params: np.ndarray
    ds: np.ndarray
    ds_elems: np.ndarray
    bpm_names: list

