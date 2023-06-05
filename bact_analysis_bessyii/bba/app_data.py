"""
So that repeated data loading is faster ...
"""

from databroker import catalog
import xarray as xr
from .preprocess_data import load_and_check_data
from bact_analysis.utils.preprocess import reorder_by_groups
import functools


@functools.lru_cache
def load_and_rearrange_data(uid: str, catalog_name: str = "heavy") -> xr.Dataset:
    """load data using uid and make it selectable per magnet
    Todo:
        Require loading from other formats?
    """
    db = catalog[catalog_name]
    run = db[uid]

    preprocessed_, dt_configuration = load_and_check_data(run, device_name="bpm")

    # ignore data that of first reading ... could be from switching
    # Todo:
    #      enumerate before how often the measurement was repeated
    #      then we do not need to rely that it was added during the
    #      measurement
    idx = preprocessed_.cs_setpoint >= 1
    preprocessed = preprocessed_.isel(time=idx)

    # make data selectedable by magnet name
    # Todo:
    #    will need to change later on ..
    #    I use here that sufficient data is available for each magnet
    reordered = reorder_by_groups(
        preprocessed,
        preprocessed.groupby(preprocessed.mux_selected_multiplexer_readback),
        reordered_dim="name",
        dim_sel="time",
        new_indices_dim="step",
    )
    rearranged = xr.concat(reordered, dim="name")
    return rearranged, dt_configuration


__all__ = ["load_and_rearrange_data"]
