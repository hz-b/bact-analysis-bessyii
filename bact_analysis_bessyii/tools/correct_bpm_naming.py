import copy as _copy
from typing import Sequence, Dict
import functools


@functools.lru_cache(maxsize=None)
def bpm_config():
    from bact_bessyii_ophyd.devices.pp.bpm_parameters import create_bpm_config
    import pandas as pd

    df = pd.DataFrame(create_bpm_config())
    df = df.set_index("name")
    return df


@functools.lru_cache(maxsize=None)
def bpm_correct_name(bpm_name):
    conf = bpm_config()
    # Find out which name was used and to which index it corresponded
    idx = int(conf.loc[bpm_name, "idx"])
    # The code was missing that minus one was subtracted
    # so the correct name should be the one which corresponds
    # to and index which is one higher
    index_of_used_entry = idx + 1
    correct_name = conf.index[index_of_used_entry == conf.idx].values
    if correct_name:
        (correct_name,) = correct_name
    else:
        correct_name = f"no_name_for_index_{index_of_used_entry}"
    return correct_name


def correct_bpm_name(bpm_data: Sequence[Dict]) -> Sequence[Dict]:
    """ """
    bpm_data = bpm_data.copy()

    def correct(bpm: Dict) -> Dict:
        n_bpm = bpm.copy()
        n_bpm["name"] = bpm_correct_name(bpm["name"])
        return n_bpm

    bpm_data = [correct(a_bpm) for a_bpm in bpm_data]
    return bpm_data


def measurement_points_bpm_data_correct_name(measurement_points):
    import copy as _copy

    def f(point):
        point = _copy.copy(point)
        point.bpm = correct_bpm_name(point.bpm)
        return point

    return [f(point) for point in measurement_points]


def measurement_per_magnet_bpm_data_correct_name(meas_per_magnet):

    meas_per_magnet = _copy.copy(meas_per_magnet)
    meas_per_magnet.per_magnet = measurement_points_bpm_data_correct_name(
        meas_per_magnet.per_magnet
    )
    return meas_per_magnet
