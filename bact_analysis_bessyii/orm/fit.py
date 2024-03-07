from dataclasses import dataclass
from typing import Sequence, Dict

import numpy as np
from scipy.linalg import lstsq

from bact_math_utils.linear_fit import x_to_cov, cov_to_std
from .model import FitResultPerBPM, FitResultPerMagnet, FitResultAllMagnets
from ..model.analysis_model import (
    FitResult,
    MeasuredValues,
    FitReadyDataPerMagnet,
    FitReadyData,
)


def get_response_one_magnet_one_bpm(
    measured_values, excitations: Sequence[float]
) -> FitResultPerBPM:
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

    return FitResultPerBPM(
        slope=FitResult(value=fitres[0][0], std=std[0]),
        offset=FitResult(value=fitres[0][1], std=std[1]),
    )


def get_response_one_magnet_plane(
    measured_values: Sequence[MeasuredValues],
    excitations: Sequence[float],
) -> Dict[str, FitResultPerBPM]:
    """

    Todo: review if all are fitted at once
    """
    return {
        bpm_name: get_response_one_magnet_one_bpm(
            [m.data[bpm_name] for m in measured_values], excitations
        )
        for bpm_name in measured_values[0].data.keys()
    }


def get_response_one_magnet(fit_ready_data: FitReadyDataPerMagnet):
    return FitResultPerMagnet(
        x=get_response_one_magnet_plane(fit_ready_data.x, fit_ready_data.excitations),
        y=get_response_one_magnet_plane(fit_ready_data.y, fit_ready_data.excitations),
    )


def get_response(fit_ready_data: FitReadyData):
    return FitResultAllMagnets(
        data={
            item.name: get_response_one_magnet(item)
            for item in fit_ready_data.per_magnet
        }
    )
