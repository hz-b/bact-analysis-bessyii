from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from .model import (
    OrbitResponseMatrices,
    OrbitResponseMatrixPlane,
    FitResultAllMagnets,
    OrbitResponseMatricesPerSteererPlane,
)
from ..interfaces.element_families import ElementFamilies


def extract_response_matrices(
    data: FitResultAllMagnets, magnet_names
) -> OrbitResponseMatrices:
    """

    Warning:
        assumes that each data set contains the same
        set of magnet names a and the same set of
        bpm names
    """
    arranged_along_magnets = [data.get(name) for name in magnet_names]

    # for the time being I assume that all bpm's are available in
    # every data set
    # this prerequisite is not required for the preceeding processings
    # step, as data are treated point by point
    # with missing data
    # Todo: handle that not all bpm's are in all data sets
    bpm_names = [bpm_datum.x.slope.name for bpm_datum in arranged_along_magnets[0].data]

    # fmt: off
    return OrbitResponseMatrices(
        x=OrbitResponseMatrixPlane(
            slope=np.array([
                [datum.x.slope.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
            offset=np.array([
                [datum.x.offset.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
            steerers=magnet_names,
            bpms=bpm_names,
        ),
        y=OrbitResponseMatrixPlane(
            slope=np.array([
                [datum.y.slope.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
            offset=np.array([
                [datum.y.offset.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
            steerers=magnet_names,
            bpms=bpm_names,
        ),
    )
    # fmt: on


def extract_response_matrices_per_steerers(
    data: FitResultAllMagnets,
        *,
    horizontal_steerer_names: Sequence[str],
    vertical_steerer_names: Sequence[str],
) -> OrbitResponseMatricesPerSteererPlane:

    return OrbitResponseMatricesPerSteererPlane(
        horizontal_steerers=extract_response_matrices(data, horizontal_steerer_names),
        vertical_steerers=extract_response_matrices(data, vertical_steerer_names),
    )


def stack_response_submatrices(orms: OrbitResponseMatricesPerSteererPlane) -> ArrayLike:
    return np.vstack(
        [
            np.hstack(
                [
                    orms.horizontal_steerers.x.slope,
                    orms.horizontal_steerers.y.slope,
                ]
            ),
            np.hstack(
                [
                    orms.vertical_steerers.x.slope,
                    orms.vertical_steerers.y.slope,
                ]
            ),
        ]
    )
