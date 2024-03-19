import functools
import logging

from databroker import catalog

from bact_analysis.utils import preprocess
import tqdm
from bact_analysis_bessyii.model.analysis_model import MeasurementData
from bact_analysis_bessyii.model.analysis_util import get_measurement_per_magnet

#: variables with bpm names
bpm_variables = (
    "bpm_elem_data",
    "bpm_ds",
)

logger = logging.getLogger("bact-analysis")


def replaceable_dims_bpm(dataset, variable_names=bpm_variables, **kwargs) -> list:
    """replace names that are typically used by the BESSY II device"""
    return preprocess.replaceable_dims(dataset, variable_names, **kwargs)


def configuration(metadata, *, device_name: str = "dt") -> dict:
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

    (descriptor,) = metadata["descriptors"]
    config = descriptor["configuration"]
    dev_con = config[device_name]
    return dev_con


def load_and_check_data(
    dataset,
    metadata,
    *,
    device_name: str = "dt",
    pv_for_applied_current: str,
    pv_for_selected_magnet: str,
) -> MeasurementData:
    """Loads run data and renames dimensons containing bpm data

    Loads beam position monitor data and configuration data

    Args:
        run: a bluesky run
        dt: the name of the device (whose configuration should be retrieved
        load_all: load all dask arrays
    Return: (preprocessed, config)
        config: dictonary containing the configuration

    Consider loading only the required data arrays of the ru
    """
    config = configuration(metadata, device_name=device_name)

    # Quite a view variables contain bpm waveforme data. Preparation for
    # replacing the names with bpm names
    bpm_names = config["data"]["bpm_names"]
    bpm_dims = replaceable_dims_bpm(dataset, prefix="", expected_length=len(bpm_names))

    # bba

    # orm

    # Find out: repetition of measurement at this stage
    muxer_pc_current_change = preprocess.enumerate_changed_value(
        dataset[pv_for_applied_current]
    )

    muxer_pc_current_change.name = "muxer_pc_current_change"
    # Find out:
    muxer_or_pc_current_change = preprocess.enumerate_changed_value_pairs(
        dataset[pv_for_applied_current], dataset[pv_for_selected_magnet]
    )
    muxer_or_pc_current_change.name = "muxer_or_pc_current_change"

    replace_dims = {dim: "bpm" for dim in bpm_dims}
    all_data = dataset.rename(replace_dims).assign_coords(bpm=list(bpm_names))

    # ignore data that of first reading ... could be from switching
    # Todo:
    #      enumerate before how often the measurement was repeated
    #      then we do not need to rely that it was added during the
    #      measurement
    idx = all_data.cs_setpoint >= 1
    all_data__ = all_data.isel(time=idx)
    # data for one magnet
    # iterate over all magnets instead of hard coded one
    preprocessed_data = MeasurementData(
        measurement=[
            get_measurement_per_magnet(
                all_data__.isel(time=all_data__[pv_for_selected_magnet] == name),
                pv_for_selected_magnet = pv_for_selected_magnet,
                pv_for_applied_current = pv_for_applied_current,
            )
            for name in set(all_data__[pv_for_selected_magnet].values)
        ]
    )
    # flatten = flatten_for_fit(preprocessed_data.measurement[0])
    return preprocessed_data


@functools.lru_cache
def load_and_rearrange_data(
    uid: str,
    catalog_name: str = "heavy_local",
        *,
    pv_for_applied_current,
    pv_for_selected_magnet,
    load_all: bool = True,
    read_from_file: bool = False,
    prefix: str= ""
):
    """load data using uid and make it selectable per magnet
    Todo:
        Require loading from other formats?
    """
    if read_from_file:
        ds, metadata = load_data_metadata_from_files(uid, prefix=prefix)
    else:
        ds, metadata = load_data_metadata_from_catalog(uid, catalog_name=catalog_name, load_all=load_all)
    return load_and_check_data(ds, metadata, device_name="bpm",
                                   pv_for_applied_current = pv_for_applied_current ,
                                   pv_for_selected_magnet = pv_for_selected_magnet,
                                   )


def load_data_metadata_from_catalog(uid: str, catalog_name: str, *, load_all: bool):
    try:
        db = catalog[catalog_name]
        run = db[uid]
    except:
        logger.warning(f"using catalog name {catalog_name} uid {uid}")
        raise

    stream = run.primary
    del run

    ds = stream.to_dask()
    if load_all:
        for name, item in tqdm.tqdm(
                ds.items(),
                total=len(ds.variables),
                desc="Loading individual variables",
        ):
            item.load()
    return ds, stream.metadata


def load_data_metadata_from_files(uid: str, prefix):
    import bz2
    import json
    import xarray as xr

    filename = f"{prefix}-{uid}-metadata.json.bz2"
    with bz2.open(filename, "rt") as fp:
        metadata = json.load(fp)
    filename = f"{prefix}-{uid}-raw-data.json.bz2"
    with bz2.open(filename, "rt") as fp:
        ds = xr.Dataset.from_dict(json.load(fp))

    return ds, metadata


__all__ = ["replaceable_dims_bpm", "load_and_check_data", "load_and_rearrange_data"]
