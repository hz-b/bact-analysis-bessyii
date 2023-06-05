"""
"""
from typing import Sequence
from bact_analysis.utils import preprocess
import tqdm
import xarray as xr

#: variables with bpm names
bpm_variables = (
    "bpm_elem_data",
    "bpm_ds",
)


def replaceable_dims_bpm(dataset, variable_names=bpm_variables, **kwargs) -> list:
    """replace names that are typically used by the BESSY II device"""
    return preprocess.replaceable_dims(dataset, variable_names, **kwargs)


def configuration(run, *, device_name: str = "dt") -> dict:
    """Device configuration for the given run

    Loads the configuration for the specified device assuming that it
    contains only a single descriptor.

    Args:
        run: a bluesky run
        dt: the name of the device (whose configuration should be retrieved

    Return:
        config: dictonary containing the configuration

    Warning:
        loaded run must only contain a single descriptor
    """

    (descriptor,) = run.primary.metadata["descriptors"]
    configuration = descriptor["configuration"]
    dev_con = configuration[device_name]
    return dev_con


def load_and_check_data(run, *, device_name: str = "dt") -> (xr.Dataset, dict):
    """Loads run data and renames dimensons containing bpm data

    Loads beam position monitor data and configuration data

    Args:
        run: a bluesky run
        dt: the name of the device (whose configuration should be retrieved

    Return: (preprocessed, config)
        config: dictonary containing the configuration

    Consider loading only the required data arrays of the ru
    """
    all_data_ = run.primary.to_dask()
    for name, item in tqdm.tqdm(
        all_data_.items(),
        total=len(all_data_.variables),
        desc="Loading individual variables",
    ):
        item.load()

    config = configuration(run, device_name=device_name)

    # Quite a view variables contain bpm waveforme data. Preparation for
    # replacing the names with bpm names
    bpm_names = config["data"]["bpm_names"]
    bpm_dims = replaceable_dims_bpm(
         all_data_, prefix="", expected_length=len(bpm_names)
    )
    # Find out: repetition of measurement at this stage
    muxer_pc_current_change = preprocess.enumerate_changed_value(
        all_data_.mux_power_converter_setpoint
    )
    # Find out:
    muxer_pc_current_change.name = "muxer_pc_current_change"
    muxer_or_pc_current_change = preprocess.enumerate_changed_value_pairs(
        all_data_.mux_power_converter_setpoint, all_data_.mux_selected_multiplexer_readback
    )
    muxer_or_pc_current_change.name = "muxer_or_pc_current_change"

    replace_dims = {dim: "bpm" for dim in bpm_dims}
    all_data = all_data_.rename(replace_dims).assign_coords(bpm=list(bpm_names))
    preprocessed = xr.merge(
        [all_data, muxer_pc_current_change, muxer_or_pc_current_change]
    )
    return preprocessed, config


__all__ = ["replaceable_dims_bpm", "load_and_check_data"]
