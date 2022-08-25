from bact_math_utils.tune import tune_change
from bact_analysis.bba.calc import angle_to_offset as a2og
import bact2_bessyii.magnets
import xarray as xr
import functools
from typing import Sequence


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


def angles_to_offset_all(angles: xr.DataArray, *, names: Sequence, tf_scale: float=1.0):
    """

    angles are assumed to exist for the names both planes and results and errors
    """
    calib = load_calib_data()

    hw2phys = calib.hw2phys.sel(name=names)
    length = calib.length.sel(name=names)
    # hw2phys contains already the polarity
    polarity = 1

    angle_scale = angles.orbit.attrs["theta"]

    def f(angle):
        # def angle_to_offset(tf: float, length: float, polarity: int, alpha: float) -> float:
        t_length = length
        # t_length = 1
        return a2og(hw2phys, t_length, polarity, angle * angle_scale, tf_scale=tf_scale)

    sel = angles.fit_params.sel(name=names)
    x = f(sel.sel(parameter="scaled_angle", result="value", plane="x"))
    y = f(sel.sel(parameter="scaled_angle", result="value", plane="y"))
    x_err = f(sel.sel(parameter="scaled_angle", result="error", plane="x"))
    y_err = f(sel.sel(parameter="scaled_angle", result="error", plane="y"))

    res = xr.DataArray(
        data=[[x, x_err], [y, y_err]],
        dims=["plane", "result", "name"],
        coords=[["x", "y"], ["value", "error"], names],
    )
    return res
