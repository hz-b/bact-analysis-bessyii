import numpy as np

from .model import (
    OrbitResponseBPMs,
    OrbitResponseSubmatrix,
    FitResultAllMagnets,
    OrbitResponseMatrices,
    OrbitResponseMatrixPlane,
)


def extract_matrix(data, magnet_names) -> OrbitResponseBPMs:
    def extract(datum):
        (r,) = datum
        return r

    arranged_along_magnets = [
        extract([datum for datum in data.data if datum.name == name])
        for name in magnet_names
    ]

    # fmt: off
    return OrbitResponseBPMs(
        x=OrbitResponseSubmatrix(
            slope=np.array([
                [datum.x.slope.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
            offset=np.array([
                [datum.x.offset.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
        ),
        y=OrbitResponseSubmatrix(
            slope=np.array([
                [datum.y.slope.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
            offset=np.array([
                [datum.y.offset.value for datum in row.data]
                for row in arranged_along_magnets
            ]),
        ),
    )
    # fmt: on


def extract_matrices(data: FitResultAllMagnets) -> OrbitResponseMatrices:
    horizontal_steerer_names = [
        datum.name for datum in data.data if datum.name[0] == "H"
    ]
    vertical_steerer_names = [datum.name for datum in data.data if datum.name[0] == "V"]

    # for the time being I assume that all bpm's are available in
    # every data set
    # this prerequisite is not required for the preceeding processings
    # step, as data are treated point by point
    # with missing data
    # Todo: handle that not all bpm#s are in all data sets
    bpm_names = [datum.name for datum in data.data[0].data]

    return OrbitResponseMatrices(
        horizontal_steerers=OrbitResponseMatrixPlane(
            matrix=extract_matrix(data, horizontal_steerer_names),
            steerers=horizontal_steerer_names,
            bpms=bpm_names,
        ),
        vertical_steerers=OrbitResponseMatrixPlane(
            matrix=extract_matrix(data, vertical_steerer_names),
            steerers=vertical_steerer_names,
            bpms=bpm_names,
        ),
    )
