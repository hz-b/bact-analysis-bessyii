from .model import OrbitResponseMatrixPlane, OrbitResponseMatrices

from enum import Enum
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


class PlaneName(Enum):
    x = "x"
    y = "y"


def plot_one_orm(axis, orm: OrbitResponseMatrixPlane, plane: str):

    PlaneName(plane)
    x = np.arange(len(orm.bpms))
    y = np.arange(len(orm.steerers))

    X, Y = np.meshgrid(x, y)
    surf = axis.plot_surface(
        X,
        Y,
        getattr(orm.matrix, plane).slope,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    return surf


def plot_orms(orms: OrbitResponseMatrices):
    def set_xy_axis_ticks(ax, bpm_names, steerer_names):
        ax.set_xticks(np.arange(len(bpm_names)))
        ax.set_xticklabels(bpm_names)
        ax.set_yticks(np.arange(len(steerer_names)))
        ax.set_yticklabels(steerer_names)

        for labels in [ax.xaxis.get_ticklabels(), ax.yaxis.get_ticklabels()]:
            plt.setp(labels, fontsize="xx-small")
            plt.setp(labels, rotation=45)

    for orm in [orms.horizontal_steerers, orms.vertical_steerers]:
        for plane in ["x", "y"]:
            fig = plt.figure(figsize=plt.figaspect(1))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            surf = plot_one_orm(ax, orms.horizontal_steerers, "x")
            ax.set_xlabel(f"bpm: {plane}")
            ax.set_ylabel(f"steerer: {plane}")
            set_xy_axis_ticks(ax, orm.bpms, orm.steerers)
