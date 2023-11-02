import functools
import logging
from bact_analysis.utils import preprocess
from bact_analysis_bessyii.model.analysis_model import (
    MeasurementPerMagnet,
    MeasurementPoint,
    FitReadyDataPerMagnet,
    MeasuredItem,
    MeasuredValues,
    BPMCalibration,
    BPMCalibrationPlane,
)
from typing import List, Sequence, Dict
from collections import OrderedDict as OrderedDictmpl
from abc import abstractmethod, ABCMeta

logger = logging.getLogger("bact-analysis")


def get_measurement_per_magnet(data_for_one_magnet):
    # todo: validate that setpoint and readback are within limits
    (name,) = set(data_for_one_magnet.mux_selected_multiplexer_readback.values)

    muxer_or_pc_current_change = preprocess.enumerate_changed_value_pairs(
        data_for_one_magnet.mux_power_converter_setpoint,
        data_for_one_magnet.mux_selected_multiplexer_readback,
    )

    return MeasurementPerMagnet(
        name=name,
        # Bluesky stacks ups the measurement on this time axis
        per_magnet=[
            get_measurement_point(data_for_one_magnet.sel(time=t), step=step)
            for t, step in zip(
                data_for_one_magnet.coords["time"].values,
                muxer_or_pc_current_change.values,
            )
        ],
    )


def flatten_for_fit(
    magnet_measurement_data: MeasurementPerMagnet, magnet_name, *, pos="pos", rms="rms"
) -> FitReadyDataPerMagnet:
    x_values = []
    y_values = []
    # flatten out the measurement points
    for measurement_point in magnet_measurement_data:
        x_data = []
        y_data = []

        # flatten out the bpm data
        for bpm in measurement_point.bpm:
            x_data.append(MeasuredItem(bpm["x"][pos], bpm["x"][rms]))
            y_data.append(MeasuredItem(bpm["y"][pos], bpm["y"][rms]))

        # data of the bpm: flattened out
        # fmt:off
        x_values.append(MeasuredValues(OrderedDictmpl(zip([bpm["name"] for bpm in measurement_point.bpm], x_data))))
        y_values.append(MeasuredValues(OrderedDictmpl(zip([bpm["name"] for bpm in measurement_point.bpm], y_data))))
        # fmt:on

    return FitReadyDataPerMagnet(
        name=magnet_name,
        steps=[measurement_point.step for measurement_point in magnet_measurement_data],
        excitations=[
            measurement_point.excitation
            for measurement_point in magnet_measurement_data
        ],
        x=x_values,
        y=y_values,
        bpm_pos=None,  # todo: add bpm_pos measurement_point.bpm_pos | None
    )


def get_measurement_point(magnet_data_per_point, *, step):
    # extact bpm x and y from the data into an array
    return MeasurementPoint(
        step=step,
        excitation=float(magnet_data_per_point.mux_power_converter_setpoint.values),
        bpm=magnet_data_per_point.bpm_elem_data.values,
    )


# def flatten_sequence_of_ordered__dict_as_array():


def get_data_as_lists(
    fit_data_for_one_magnet: MeasuredValues,
) -> (List[List[float]], List[List[float]]):
    vals = [[v.value for _, v in item.data.items()] for item in fit_data_for_one_magnet]
    rms = [[v.rms for _, v in item.data.items()] for item in fit_data_for_one_magnet]
    return vals, rms


class BPMCalibrationsRepository(metaclass=ABCMeta):
    @abstractmethod
    def get(self, name: str) -> BPMCalibration:
        raise NotImplementedError("abstract base class")


def bpm_raw_data_to_m(
    a_bpm: Dict, *, copy: bool = True, calib_repo: BPMCalibrationsRepository
) -> Dict:
    """

    Assuming that it is not needed on the long run
    Conversions will be required every now and then

    Todo:
        to which module does this code belong to?

        avoid side effect return a new copy by default
    """
    assert copy
    if copy:
        a_bpm = a_bpm.copy()

    c = calib_repo.get(a_bpm["name"])
    for plane in ["x", "y"]:
        cp = getattr(c, plane)
        # add pos and rms to the data
        d = a_bpm[plane]
        if copy:
            d = d.copy()
        d["pos"] = cp.to_pos(d["pos_raw"])
        d["rms"] = cp.to_rms(d["rms_raw"])
        if copy:
            a_bpm[plane] = d
    return a_bpm


def bpms_raw_data_to_m(
    bpm_data: Sequence[Dict],
    *,
    copy: bool = True,
    calib_repo: BPMCalibrationsRepository
) -> Sequence[Dict]:
    """ """
    assert copy
    if copy:
        bpm_data = bpm_data.copy()

    def bpm_is_active(bpm_name):
        """at least one channel"""
        c = calib_repo.get(bpm_name)
        flag = c.x.active | c.y.active
        if not flag:
            logger.warning("Ignoring bpm %s", bpm_name)
        return flag

    bpm_data = [
        bpm_raw_data_to_m(a_bpm, copy=copy, calib_repo=calib_repo)
        for a_bpm in bpm_data
        if bpm_is_active(a_bpm["name"])
    ]
    return bpm_data


def measurement_points_bpms_raw_data_to_m(
    measurement_points, *, calib_repo: BPMCalibrationsRepository
):
    import copy as _copy

    def f(point):
        point = _copy.copy(point)
        point.bpm = bpms_raw_data_to_m(point.bpm, calib_repo=calib_repo)
        return point

    return [f(point) for point in measurement_points]


def measurement_per_magnet_bpms_raw_data_to_m(
    meas_per_magnet, calib_repo: BPMCalibrationsRepository
):
    import copy as _copy

    meas_per_magnet = _copy.copy(meas_per_magnet)
    meas_per_magnet.per_magnet = measurement_points_bpms_raw_data_to_m(
        meas_per_magnet.per_magnet, calib_repo=calib_repo
    )
    return meas_per_magnet
