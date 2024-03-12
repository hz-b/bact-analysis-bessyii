"""Create reponse matrix plot using pyvista
"""
from .model import OrbitResponseMatrices
import numpy as np
import pyvista as pv
from matplotlib import colormaps


def plot_orms(orms: OrbitResponseMatrices, scale_bpm_readings=1e3, scale_offset_plot=1):
    """

    Todo:
        color should be done by offset or bpm readings
        deviation should be according to offset from prediction
    """
    # fmt: off
    X, Y = np.meshgrid(
        np.arange(
            len(orms.horizontal_steerers.bpms) + len(orms.vertical_steerers.bpms)
        ),
        np.arange(
            len(orms.horizontal_steerers.steerers) + len(orms.vertical_steerers.steerers)
        ),
    )
    Z = np.vstack([
            np.hstack([
                orms.horizontal_steerers.matrix.x.slope,
                orms.horizontal_steerers.matrix.y.slope,
            ]),
            np.hstack([
                orms.vertical_steerers.matrix.x.slope,
                orms.vertical_steerers.matrix.y.slope,
            ]),
        ])
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
