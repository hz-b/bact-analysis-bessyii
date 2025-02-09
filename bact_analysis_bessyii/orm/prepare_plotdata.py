from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from .model import (
    OrbitResponseMatrices,
    OrbitResponseMatrixPlane,
    FitResultAllMagnets,
    OrbitResponseMatricesPerSteererPlane,
)


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
    bpm_names = [bpm_datum.name for bpm_datum in arranged_along_magnets[0].data]

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
) -> OrbitResponseMatricesPerSteererPlane:
    horizontal_steerer_names = [
        datum.name for datum in data.data if datum.name[0] == "H"
    ]
    vertical_steerer_names = [datum.name for datum in data.data if datum.name[0] == "V"]

    def reponse_mat(steerer_names: Sequence[str]):
        if len(steerer_names) > 0:
            return extract_response_matrices(data, steerer_names)
        else:
            return None

    h_mat, v_mat = [
        reponse_mat(steerer_names)
        for steerer_names in (horizontal_steerer_names, vertical_steerer_names)
    ]

    return OrbitResponseMatricesPerSteererPlane(
        horizontal_steerers=h_mat, vertical_steerers=v_mat
    )


def stack_response_submatrices(orms: OrbitResponseMatricesPerSteererPlane) -> ArrayLike:
    assert orms.horizontal_steerers
    assert orms.vertical_steerers
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
