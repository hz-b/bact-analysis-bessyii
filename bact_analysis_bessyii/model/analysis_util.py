import logging
from bact_analysis.utils import preprocess
from bact_analysis_bessyii.model.analysis_model import MeasurementPerMagnet, MeasurementPoint, FitReadyDataPerMagnet, \
    MeasuredItem, MeasuredValues
from typing import List
from collections import OrderedDict as OrderedDictmpl
logger = logging.getLogger("bact-analysis")

def get_measurement_per_magnet(data_for_one_magnet):
    # todo: validate that setpoint and readback are within limits
    name, = set(data_for_one_magnet.mux_selected_multiplexer_readback.values)

    muxer_or_pc_current_change = preprocess.enumerate_changed_value_pairs(
        data_for_one_magnet.mux_power_converter_setpoint,
        data_for_one_magnet.mux_selected_multiplexer_readback
    )

    return MeasurementPerMagnet(
        name = name,
        # Bluesky stacks ups the measurement on this time axis
        per_magnet=[get_measurement_point(data_for_one_magnet.sel(time=t), step=step)
                    for t, step  in zip(data_for_one_magnet.coords['time'].values, muxer_or_pc_current_change.values)
                    ]
    )

def flatten_for_fit(magnet_measurement_data:MeasurementPerMagnet) -> FitReadyDataPerMagnet:

    x_values = []
    y_values = []
    # flatten out the measurement points
    for measurement_point in magnet_measurement_data.per_magnet:
        x_data = []
        y_data = []

        # flatten out the bpm data
        for bpm in measurement_point.bpm:
            x_data.append(MeasuredItem(bpm["x"]["pos_raw"], bpm["x"]["rms_raw"]))
            y_data.append(MeasuredItem(bpm["y"]["pos_raw"], bpm["y"]["rms_raw"]))


        # data of the bpm: flattened out
        x_values.append(
            MeasuredValues(OrderedDictmpl(zip([bpm["name"] for bpm in measurement_point.bpm], x_data)))
        )
        y_values.append(
            MeasuredValues(OrderedDictmpl(zip([bpm["name"] for bpm in measurement_point.bpm], y_data)))
        )

    return FitReadyDataPerMagnet(
        name=magnet_measurement_data.name,
        steps=[measurement_point.step for measurement_point in magnet_measurement_data.per_magnet],
        excitations=[measurement_point.excitation for measurement_point in magnet_measurement_data.per_magnet],
        x=x_values,
        y=y_values,
        bpm_pos=None # todo: add bpm_pos measurement_point.bpm_pos | None
    )
def get_measurement_point(magnet_data_per_point, *, step):
    # extact bpm x and y from the data into an array
    return MeasurementPoint(
        step = step,
        excitation = float(magnet_data_per_point.mux_power_converter_setpoint.values),
        bpm = magnet_data_per_point.bpm_elem_data.values
    )

# def flatten_sequence_of_ordered__dict_as_array():

def get_data_as_lists(fit_data_for_one_magnet: MeasuredValues) ->(List[List[float]], List[List[float]]):
        vals =  [[v.value for _, v in item.data.items()] for item in fit_data_for_one_magnet]
        rms =  [[v.rms for _, v in item.data.items()] for item in fit_data_for_one_magnet]
        return vals, rms


