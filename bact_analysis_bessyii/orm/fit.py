from dataclasses import dataclass
from typing import Sequence, Dict

import numpy as np
from scipy.linalg import lstsq

from bact_math_utils.linear_fit import x_to_cov, cov_to_std
from .model import FitResultPerBPM, FitResultPerMagnet, FitResultAllMagnets, FitResultBPMPlane
from ..model.analysis_model import (
    FitResult,
    MeasuredValues,
    FitReadyDataPerMagnet,
    FitReadyData, MeasuredItem,
)


def get_response_one_magnet_one_bpm_one_plane(
    measured_values: Sequence[MeasuredItem], excitations: Sequence[float]
) -> FitResultBPMPlane:
    """

    Todo:
        need to add weights / noise treatment
    Args:
        measured_values:
        excitations:

    Returns:

    """

    bpm_vals = np.array([m.value for m in measured_values])
    bpm_rms = np.array([m.rms for m in measured_values])

    A = np.vstack([excitations, np.ones(len(bpm_vals))]).T
    fitres = lstsq(A, bpm_vals)
    _, residues, rank, _ = fitres
    N, p = A.shape

    if rank != p:
        raise AssertionError(f"Fit with {p} parameters returned a rank of {rank}")

    # only works if using numpy arrays
    cov = x_to_cov(A, fitres[1], N, p)
    std = cov_to_std(cov)

    return FitResultBPMPlane(
        slope=FitResult(value=fitres[0][0], std=std[0]),
        offset=FitResult(value=fitres[0][1], std=std[1]),
    )


def get_response_one_magnet_one_bpm(
    x: Sequence[MeasuredItem],  y: Sequence[MeasuredItem], excitations: Sequence[float], bpm_name:str
) -> FitResultPerBPM:
    """
    Args:
        measured_values:
        excitations:

    Returns:

    """
    return FitResultPerBPM(
        x=get_response_one_magnet_one_bpm_one_plane(x, excitations),
        y=get_response_one_magnet_one_bpm_one_plane(y, excitations),
        name=bpm_name
    )



def get_response_one_magnet(fit_ready_data: FitReadyDataPerMagnet) -> FitResultPerMagnet:
    return FitResultPerMagnet(data=[
        get_response_one_magnet_one_bpm(
            # for all e
            x=[datum.data[bpm_name] for datum in fit_ready_data.x],
            y=[datum.data[bpm_name] for datum in fit_ready_data.y],
            excitations=fit_ready_data.excitations,
            bpm_name=bpm_name
        )
        for bpm_name in fit_ready_data.x[0].data],
        name=fit_ready_data.name
    )


def get_response(fit_ready_data: FitReadyData):
    r =  FitResultAllMagnets(
        data=[get_response_one_magnet(item)
            for item in fit_ready_data.per_magnet]
    )
    return r
