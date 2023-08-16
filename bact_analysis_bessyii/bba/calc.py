from bact_math_utils.tune import tune_change
from bact_analysis.bba.calc import angle_to_offset as a2og
import bact2_bessyii.magnets
import xarray as xr
import functools
from typing import Sequence
from enum import IntEnum
import numpy as np


@functools.lru_cache(maxsize=1)
def load_calib_data():
    df = bact2_bessyii.magnets.quadrupole_calbration_factors_mongodb()
    calib = (
        xr.Dataset.from_dataframe(df)
        .rename_dims(index="name")
        .assign_coords(name=df.index)
    )
    return calib


def predict_tune_change(
    name: str, dI: float, *, beta: float, f: float = 500e6, nb: int = 400
) -> float:
    """Use betatron function to predict tune change
    """
    calib = load_calib_data()

    hw2phys = calib.hw2phys.sel(name=name)
    length = calib.length.sel(name=name)

    dk = hw2phys * dI
    dT = tune_change(dk, beta, length=length, f=f, nb=nb)
    return dT


def angle_to_offset(angles, *, names):

    raise NotImplementedError("Untested code below")
    calib = load_calib_data()

    hw2phys = calib.hw2phys.sel(name=names)
    length = calib.length.sel(name=names)
    # hw2phys contains already the polarity
    polarity = 1
    res = a2og(hw2phys, length, polarity, angle)

    return res



quad_length = dict(
    # mechanical length: Q4
    Q4=0.5,
    # mechanical length: Q1
    Q1=0.25,
    # mechanical length: Q2
    Q2=0.2,
    # mechanical length: Q3
    Q3=0.25,
    # mechanical length: Q5
    Q5=0.2,
)

class Polarity(IntEnum):
    pos = 1
    neg = -1

quad_polarities = dict(
    # horizontal focusing
    Q1=Polarity.pos,
    # vertical focusing
    Q2=Polarity.neg,
    # vertical focusing
    Q3=Polarity.neg,
    # horizontal focusing
    Q4=Polarity.pos,
    # vertical focusing
    Q5=Polarity.neg,
)





# from Peter's hand note
# this value is an average transfer function for the
# auxilliary coil winding for the muxer
# todo: leave it only here for cross check at the current development stage
#       needs to go to report
delta_g = 0.796 / 5.0
brho = 5.67044
delta_k = delta_g / brho


def angles_to_offset_all(angles: xr.DataArray, *, names: Sequence, tf_scale: float=1.0):
    """

    angles are assumed to exist for the names both planes and results and errors
    """
    # calib = load_calib_data()

    # hw2phys = calib.hw2phys.sel(name=names)
    # length = calib.length.sel(name=names)
    # hw2phys contains already the polarity
    # polarity = 1
    #TODO:  rework and store the data in db
    polarities = np.array([quad_polarities[name[:2].upper() ]for name in np.array([s.capitalize() for s in names])])
    length = np.array([quad_length[name[:2]] for name in np.array([s.capitalize() for s in names])])
    angle_scale = angles.orbit.attrs["theta"]

    def f(angle):
        # def angle_to_offset(tf: float, length: float, polarity: int, alpha: float) -> float:
        t_length = length
        # t_length = 1
        # transfer function of the central zone ... 0.1 for nearly all magnets
        # need to look it up from database
        tf = 0.01
        return a2og(tf=tf, length=t_length, polarity=polarities, alpha=angle * angle_scale, tf_scale=tf_scale)

    sel = angles.fit_params.sel(name=names)
    x = f(sel.sel(parameter="scaled_angle", result="value", plane="x"))
    y = - f(sel.sel(parameter="scaled_angle", result="value", plane="y"))
    x_err = f(sel.sel(parameter="scaled_angle", result="error", plane="x"))
    y_err = f(sel.sel(parameter="scaled_angle", result="error", plane="y"))

    x_err = np.absolute(x_err)
    y_err = np.absolute(y_err)
    res = xr.DataArray(
        data=[[x, x_err], [y, y_err]],
        dims=["plane", "result", "name"],
        coords=[["x", "y"], ["value", "error"], names],
    )
    return res
