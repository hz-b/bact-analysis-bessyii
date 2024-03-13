import functools
import logging

from databroker import catalog
import tqdm

from bact_analysis.utils import preprocess
from ..bba.preprocess_data import (
    load_and_check_data,
    configuration,
    replaceable_dims_bpm,
)
from ..model.analysis_model import MeasurementData
from ..model.analysis_util import get_measurement_per_magnet

logger = logging.getLogger("bact-analysis")


def load_and_check_data(
    dataset, metadata, *, device_name: str = "dt"
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
    # Find out: repetition of measurement at this stage
    muxer_pc_current_change = preprocess.enumerate_changed_value(
        dataset["mux_sel_p_setpoint"]
    )
    # Find out:
    muxer_pc_current_change.name = "muxer_pc_current_change"
    muxer_or_pc_current_change = preprocess.enumerate_changed_value_pairs(
        dataset["mux_sel_p_setpoint"], dataset["mux_sel_selected"]
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
                all_data__.isel(time=all_data__["mux_sel_selected"] == name)
            )
            for name in set(all_data__["mux_sel_selected"].values)
        ]
    )
    # flatten = flatten_for_fit(preprocessed_data.measurement[0])
    return preprocessed_data


@functools.lru_cache
def load_and_rearrange_data(
    uid: str,
    catalog_name: str = "heavy_local",
    load_all: bool = True,
    read_from_file=False,
):
    """load data using uid and make it selectable per magnet
    Todo:
        Require loading from other formats?
    """
    if read_from_file:
        load_and_rearrange_data_from_files(uid)
    else:
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

        return load_and_check_data(ds, stream.metadata, device_name="bpm")
