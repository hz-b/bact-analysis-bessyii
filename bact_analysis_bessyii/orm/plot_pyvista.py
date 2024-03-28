"""Create response matrix plot using pyvista
"""
from .model import OrbitResponseMatricesPerSteererPlane
from .prepare_plotdata import stack_response_submatrices

import numpy as np
import pyvista as pv
from matplotlib import colormaps

from ..interfaces.device_location import DeviceLocationServiceInterface
from ..model.accelerator_model import AcceleratorDescription


def plot_orms(
    orms: OrbitResponseMatricesPerSteererPlane,
    acc_desc: AcceleratorDescription,
    name_pos_service: DeviceLocationServiceInterface,
    scale_bpm_readings=1e3,
    scale_offset_plot=1,

):
    """

    Todo:
        color should be done by offset or bpm readings
        deviation should be according to offset from prediction
    """
    Z = stack_response_submatrices(orms)

    def name_to_position(device_name: str) -> float:
        if device_name == "BPMZ41T6R":
            # how to deal with e.g. BPM's that are not in the survey data?
            # currently I place it artificially somewere
            return acc_desc.survey.get(
                name_pos_service.get_location_name("BPMZ4T6R")
            ).value + 0.25
        return acc_desc.survey.get(
            name_pos_service.get_location_name(device_name)
        ).value

    # assuming same bpm names valid for x and y
    # just using x for horizontal steerers and y for vertical steerers
    s_bpms_h_st = np.array([name_to_position(name) for name in orms.horizontal_steerers.x.bpms])
    s_bpms_v_st = np.array([name_to_position(name) for name in orms.vertical_steerers.y.bpms])
    s_h_st = np.array([name_to_position(name) for name in orms.horizontal_steerers.x.steerers])
    s_v_st = np.array([name_to_position(name) for name in orms.vertical_steerers.y.steerers])

    # fmt: off
    X, Y = np.meshgrid(
        np.hstack([-s_bpms_h_st.max() + s_bpms_h_st, s_bpms_v_st]),
        np.hstack([-s_h_st.max() + s_h_st, s_v_st]),
    )
    # fmt: on

    Z = (Z * scale_bpm_readings).astype(np.float32)
    grid = pv.StructuredGrid(
        X.astype(np.float32),
        Y.astype(np.float32),
        (Z * scale_offset_plot).astype(np.float32),
    )
    grid["orbit_offset"] = Z.ravel()

    pl = pv.Plotter(lighting="three lights")
    pl.add_mesh(grid, cmap=colormaps["viridis"], show_scalar_bar=True)
    pl.show_bounds(
        xtitle="bpms",
        ytitle="steerers",
        ztitle="offset",
        show_zlabels=False,
        color="k",
        font_size=26,
    )
    pl.add_text("Orbit response matrix of BESSY II")
    pl.show()
