from typing import Sequence
import numpy as np
import pyvista as pv
from matplotlib import colormaps

from dt4acc.model.planes import Planes

from .model import OrbitPredictionCollection, AcceleratorDescription, OrbitPredictionForKicks, OrbitPredictionForPlane
from ..model.analysis_model import FitReadyData, EstimatedAngles, MagnetEstimatedAngles, ErrorType, \
    FitReadyDataPerMagnet, MeasuredValues, FitResult


def get_orbit_maximum_offset(orbits_collection: Sequence[OrbitPredictionForPlane]):
    """for each orbit/beam position return maximum seen offset from orbit"""
    tmp = np.array([
        [orbit.value for orbit in orbits.orbit]
        for orbits in orbits_collection
    ])
    return np.max(np.absolute(tmp), axis=0)

def get_difference_measurements_orbits_for_magnet(
            measured_data: FitReadyDataPerMagnet, orbits: OrbitPredictionForKicks, *, plane: Planes
):
    return [
        get_difference_measurements_orbits_for_magnet_for_excitation(m, orb)
        for m, orb in zip(measured_data.get(plane), orbits.get(plane))
    ]

def get_difference_measurements_orbits_for_magnet_for_excitation(
    measurement: MeasuredValues, orbit: OrbitPredictionForPlane
):
    orb = np.array([datum.value for datum in orbit.orbit])
    mea = [datum.value for datum in measurement.data]
    return mea - orb


def plot_forecast_difference_3D(
        measured_data: FitReadyData,
        orbit_prediction: OrbitPredictionCollection,
        fits: EstimatedAngles,
        acc_desc: AcceleratorDescription,
        orbit_scale: float=1e4
):

    # assuming same bpm names valid for x and y
    bpm_names = [datum.name for datum in measured_data.per_magnet[0].x[0].data]

    s_bpms = [acc_desc.survey.get(name).value for name in bpm_names]

    steerer_names = [datum.name for datum in measured_data.per_magnet]
    # s_steerers = [acc_desc.survey.get(name).value for name in steerer_names]

    # some steerers have not as many excitations as the others
    difference_measurements_orbits_per_excitations_x = [
        get_difference_measurements_orbits_for_magnet(m, o, plane=Planes.x)
        for m, o in zip(measured_data.per_magnet, orbit_prediction.per_magnet)
    ]

    # some steerers have not as many excitations as the others
    difference_measurements_orbits_per_excitations_y = [
        get_difference_measurements_orbits_for_magnet(m, o, plane=Planes.y)
        for m, o in zip(measured_data.per_magnet, orbit_prediction.per_magnet)
    ]

    x_mae = np.array([
        np.mean(np.array(d), axis=0) for d in difference_measurements_orbits_per_excitations_x
    ], dtype=float)

    y_mae = np.array([
        np.mean(np.array(d), axis=0) for d in difference_measurements_orbits_per_excitations_y
    ], dtype=float)

    x_mse = np.array(
        [
            np.sum(np.array(d) ** 2, axis=0)
            for d in difference_measurements_orbits_per_excitations_x
        ]
    )

    y_mse = np.array([
            np.sum(np.array(d) ** 2, axis=0)
            for d in difference_measurements_orbits_per_excitations_x
    ])


    x_max_excitation = np.array(
        [get_orbit_maximum_offset(orbits.x) for orbits in orbit_prediction.per_magnet]
    )

    y_max_excitation = np.array(
        [get_orbit_maximum_offset(orbits.y) for orbits in orbit_prediction.per_magnet]
    )

    X, Y = np.meshgrid(s_bpms, range(len(steerer_names)))
    # fmt: on

    grid_x = pv.StructuredGrid(
        X.astype(np.float32),
        Y.astype(np.float32),
        (x_max_excitation).astype(np.float32) * orbit_scale
    )
    grid_x["mae"] = x_mae.ravel()
    grid_x["mse"] = x_mse.ravel()

    grid_y = pv.StructuredGrid(
        X.astype(np.float32),
        Y.astype(np.float32),
        (x_max_excitation).astype(np.float32) * orbit_scale
    )
    grid_y["y_mae"] = y_mae.ravel()
    grid_y["y_mse"] = y_mse.ravel()

    pl = pv.Plotter(lighting="three lights")
    pl.add_mesh(grid_x, cmap=colormaps["viridis"], show_scalar_bar=True)
    pl.show_bounds(
        xtitle="bpms",
        ytitle="steerers",
        ztitle="offset",
        show_zlabels=False,
        color="k",
        font_size=26,
    )
    pl.add_text("Steerer response: mse")
    pl.show()
