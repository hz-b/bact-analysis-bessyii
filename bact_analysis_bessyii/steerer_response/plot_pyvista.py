from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike
import pyvista as pv
from matplotlib import colormaps

from dt4acc.model.planes import Planes

from .model import (
    OrbitPredictionCollection,
    AcceleratorDescription,
    OrbitPredictionForKicks,
    OrbitPredictionForPlane,
)
from ..interfaces.device_location import DeviceLocationServiceInterface
from ..interfaces.element_families import ElementFamilies
from ..model.analysis_model import (
    FitReadyData,
    EstimatedAngles,
    MagnetEstimatedAngles,
    ErrorType,
    FitReadyDataPerMagnet,
    MeasuredValues,
    FitResult,
)


@dataclass
class DifferencesPerMagnets:
    per_magnet: Sequence[ArrayLike]

    @property
    def mean_square_error(self):
        return np.array([np.sum(np.array(d) ** 2, axis=0) for d in self.per_magnet])

    @property
    def mean_absolute_error(self):
        return np.array([np.mean(np.array(d), axis=0) for d in self.per_magnet], dtype=float)


@dataclass
class MaximumOrbitOffsetPerMagnet:
    data: ArrayLike


@dataclass
class SteererResponseResultPerMagnet:
    diff: DifferencesPerMagnets
    maximum_offset: MaximumOrbitOffsetPerMagnet


def plot_forecast_difference_3d(
    measured_data: FitReadyData,
    orbit_prediction: OrbitPredictionCollection,
    fits: EstimatedAngles,
    acc_desc: AcceleratorDescription,
    element_families: ElementFamilies,
    name_pos_service: DeviceLocationServiceInterface,
    orbit_scale: float = 1e4,
):
    def name_to_position(device_name: str) -> float:
        return acc_desc.survey.get(name_pos_service.get_location_name(device_name)).value

    # assuming same bpm names valid for x and y
    bpm_names = [datum.name for datum in measured_data.per_magnet[0].x[0].data]
    s_bpms = np.array([name_to_position(name) for name in bpm_names])

    steerer_names = [datum.name for datum in measured_data.per_magnet]
    # limit to known ones

    h_steerers = [name for name in steerer_names if name in element_families.get("horizontal_steerers").members]
    v_steerers = [name for name in steerer_names if name in element_families.get("vertical_steerers").members]

    h_steerers.sort(key=name_to_position)
    v_steerers.sort(key=name_to_position)

    s_h_steerers = np.array([name_to_position(name) for name in h_steerers])
    s_v_steerers = np.array([name_to_position(name) for name in v_steerers])

    def extract_steerer_response(magnet_names, plane):
        return get_difference_measurements_orbits_for_magnets_plane_per_excitation(
            measured_data, orbit_prediction, magnet_names, plane
        )

    # some steerers have not as many excitations as the others
    h_st_x = extract_steerer_response(h_steerers, "x")
    h_st_y = extract_steerer_response(h_steerers, "y")
    v_st_x = extract_steerer_response(v_steerers, "x")
    v_st_y = extract_steerer_response(v_steerers, "y")


    X, Y = np.meshgrid(
        np.hstack([-s_bpms.max()+s_bpms, s_bpms]),
        np.hstack([-s_h_steerers.max() +s_h_steerers, s_v_steerers])
    )

    # fmt: off
    mae = np.vstack([
        np.hstack([h_st_x.diff.mean_absolute_error, h_st_y.diff.mean_absolute_error]),
        np.hstack([v_st_x.diff.mean_absolute_error, v_st_y.diff.mean_absolute_error])
    ])
    mse = np.vstack([
        np.hstack([h_st_x.diff.mean_square_error, h_st_y.diff.mean_square_error]),
        np.hstack([v_st_x.diff.mean_square_error, v_st_y.diff.mean_square_error])
    ])

    max_off = np.vstack([
        np.hstack([h_st_x.maximum_offset.data, h_st_y.maximum_offset.data]),
        np.hstack([v_st_x.maximum_offset.data, v_st_y.maximum_offset.data]),
    ])
    # fmt: on


    grid = pv.StructuredGrid(
        X.astype(np.float32),
        Y.astype(np.float32),
        max_off.astype(np.float32) * orbit_scale
    )
    grid["mae"] = mae.ravel()
    grid["mse"] = mse.ravel()

    pl = pv.Plotter(lighting="three lights")
    pl.add_mesh(grid, cmap=colormaps["viridis"], show_scalar_bar=True, scalars="mse")
    pl.show_bounds(
        xtitle="bpms",
        ytitle="steerers",
        ztitle="offset",
        show_zlabels=False,
        color="k",
        font_size=26,
    )
    pl.add_text("Steerer response: mse^(-1/2)")
    pl.show()


def get_difference_measurements_orbits_for_magnets_plane_per_excitation(
        measured_data: FitReadyData,
        orbit_prediction: OrbitPredictionCollection,
        magnet_names: Sequence[str],
        plane: Planes
) -> SteererResponseResultPerMagnet:
    return SteererResponseResultPerMagnet(
        diff=DifferencesPerMagnets(per_magnet=[
            get_difference_measurements_orbits_for_magnet(
                measured_data.get(magnet_name),
                orbit_prediction.get(magnet_name),
                plane=plane
            )
            for magnet_name in magnet_names
        ]),
        maximum_offset=MaximumOrbitOffsetPerMagnet(
            data=get_orbit_maximum_offsets(orbit_prediction, magnet_names, plane)
        )
    )


def get_orbit_maximum_offsets(
        orbit_prediction: OrbitPredictionCollection,
        magnet_names: Sequence[str],
        plane: Planes,
):
    return np.array([
        get_orbit_maximum_offset_per_magnet_and_plane(orbit_prediction.get(name).get(plane))
        for name in magnet_names
    ])


def get_orbit_maximum_offset_per_magnet_and_plane(orbits_collection: Sequence[OrbitPredictionForPlane]):
    """for each orbit/beam position return maximum seen offset from orbit"""
    tmp = np.array(
        [[orbit.value for orbit in orbits.orbit] for orbits in orbits_collection]
    )
    return np.max(np.absolute(tmp), axis=0)


def get_difference_measurements_orbits_for_magnet(
    measured_data: FitReadyDataPerMagnet,
    orbits: OrbitPredictionForKicks,
    *,
    plane: Planes
) -> Sequence[ArrayLike]:
    return [
        get_difference_measurements_orbits_for_magnet_for_excitation(m, orb)
        for m, orb in zip(measured_data.get(plane), orbits.get(plane))
    ]


def get_difference_measurements_orbits_for_magnet_for_excitation(
    measurement: MeasuredValues, orbit: OrbitPredictionForPlane
) -> ArrayLike:
    orb = np.array([datum.value for datum in orbit.orbit])
    mea = [datum.value for datum in measurement.data]
    return mea - orb
