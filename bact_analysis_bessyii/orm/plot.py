from typing import Sequence

from bact_analysis_bessyii.orm.model import FitResultAllMagnets
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass


@dataclass
class OrbitResponseSubmatrix:
    #: todo: add names of steerers and bpos
    slope: ArrayLike
    offset: ArrayLike

@dataclass
class OrbitResponseBPMs:
    x : OrbitResponseSubmatrix
    y : OrbitResponseSubmatrix


@dataclass
class OrbitResponseSteeres:
    x: OrbitResponseBPMs
    y: OrbitResponseBPMs


def extract_matrix(data, magnet_names) -> OrbitResponseBPMs:
    return OrbitResponseBPMs(
        x=OrbitResponseSubmatrix(
            slope=np.array([[d.slope.value for _, d in data.data[name].x.items()] for name in magnet_names]),
            offset=np.array([[d.slope.value for _, d in data.data[name].x.items()] for name in magnet_names]),
        ),
        y=OrbitResponseSubmatrix(
            slope=np.array([[d.slope.value for _, d in data.data[name].y.items()] for name in magnet_names]),
            offset=np.array([[d.slope.value for _, d in data.data[name].y.items()] for name in magnet_names]),
        )
    )


def plot_one_orm(axis, orm, steerer_names: Sequence[str], bpm_names: Sequence[str]):
    x = np.arange(len(bpm_names))
    y = np.arange(len(steerer_names))

    X, Y = np.meshgrid(x, y)
    surf = axis.plot_surface(X, Y, orm, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    return surf

def plot_orm(data: FitResultAllMagnets):
    horizontal_steerer_names = [name for name in data.data.keys() if name[0] == "H"]
    vertical_steerer_names = [name for name in data.data.keys() if name[0] == "V"]

    # for the time being I assume that all bpm's are available in any data set
    # this prerequiste has not been before, up to now it should be able to get away
    # with missing data
    # Todo: handle that not all bpm#s are in all data sets
    bpm_names = list(data.data[next(iter(data.data))].x.keys())
    matrices_horizontal_steerers = extract_matrix(data, horizontal_steerer_names)
    matrices_vertical_steerers = extract_matrix(data, vertical_steerer_names)

    def set_xy_axis_ticks(ax, steerer_names):
        ax.set_xticks(np.arange(len(bpm_names)))
        ax.set_xticklabels(bpm_names)
        ax.set_yticks(np.arange(len(steerer_names)))
        ax.set_yticklabels(steerer_names)

        for labels in [ax.xaxis.get_ticklabels(), ax.yaxis.get_ticklabels()]:
            plt.setp(labels, fontsize="xx-small")
            plt.setp(labels, rotation=45)

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = plot_one_orm(ax,  matrices_horizontal_steerers.x.slope,  horizontal_steerer_names, bpm_names)
    ax.set_xlabel("bpm: x")
    ax.set_ylabel("steerer: x")
    set_xy_axis_ticks(ax, horizontal_steerer_names)

    fig = plt.figure(figsize=plt.figaspect(1))
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = plot_one_orm(ax,  matrices_horizontal_steerers.y.slope,  horizontal_steerer_names, bpm_names)
    ax.set_xlabel("bpm: y")
    ax.set_ylabel("steerer: x")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    set_xy_axis_ticks(ax, horizontal_steerer_names)

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = plot_one_orm(ax,  matrices_vertical_steerers.x.slope,  vertical_steerer_names, bpm_names)
    ax.set_xlabel("bpm: x")
    ax.set_ylabel("steerer: y")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    set_xy_axis_ticks(ax, vertical_steerer_names)

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = plot_one_orm(ax,  matrices_vertical_steerers.y.slope,  vertical_steerer_names, bpm_names)
    ax.set_xlabel("bpm: y")
    ax.set_ylabel("steerer: y")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    set_xy_axis_ticks(ax, vertical_steerer_names)

