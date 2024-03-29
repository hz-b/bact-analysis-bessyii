import logging
import os.path
from dataclasses import asdict
from typing import Sequence
import tqdm
import xarray as xr
from bact_analysis.transverse.twiss_interpolate import interpolate_twiss
from bact_analysis.utils.preprocess import rename_doublicates, replace_names
from ..tools.preprocess_data import load_and_rearrange_data
from ..model.analysis_model import (MeasurementData, MeasurementPerMagnet, EstimatedAngles,
                                                        FitReadyData)
from ..model.analysis_util import (measurement_per_magnet_bpms_raw_data_to_m, flatten_for_fit, )
from ..model.calc import get_magnet_estimated_angle
from ..tools.correct_bpm_naming import measurement_per_magnet_bpm_data_correct_name
from bact_bessyii_mls_ophyd.db.mongo_repository import InitializeMongo
from .bpm_repo import BPMCalibrationsRepositoryBESSYII

logger = logging.getLogger("bact-analysis-bessyii")


def load_model(required_element_names: Sequence[str], filename: str = "bessyii_twiss_thor_scsi_from_twin.nc",
               datadir: str = None, ) -> Sequence[MeasurementData]:
    if datadir is None:
        datadir = os.path.dirname(__file__)
    path = os.path.join(datadir, filename)
    selected_model_ = xr.load_dataset(path)
    del selected_model_.coords["par"]

    doublets, renamed = rename_doublicates(selected_model_.coords["pos"].values.tolist())
    if len(doublets):
        logger.warning('Loaded model "%s": required to rename doubles: %s', path, doublets)
    selected_model_ = selected_model_.assign_coords(pos=renamed)

    # assume that the betatron change is small within these items ...
    # needs to be reviewed
    # Todo:
    #   element_dim is misleading: it means if data are from the start or the end of an element
    #   explain exactly what happens in the next 2 lines
    quad_twiss_ = interpolate_twiss(selected_model_, names=required_element_names)
    quad_twiss = (  # for each selected quadrupole ... go back to positiom to select for the quadrupole in question
        quad_twiss_.rename(dict(elem="pos")))
    del quad_twiss_

    n_index = replace_names(list(selected_model_.coords["pos"].values),
                            {name: name + "_s" for name in quad_twiss.coords["pos"].values}, )
    selected_model = xr.concat([selected_model_.assign_coords(pos=n_index), quad_twiss], dim="pos").sortby("ds")
    return selected_model


def get_magnet_names(preprocessed_measurement):
    # Initialize an empty list to store the names
    return [item.name for item in preprocessed_measurement.measurement if isinstance(item, MeasurementPerMagnet)]


calib_repo = BPMCalibrationsRepositoryBESSYII()


def main(uid):
    preprocessed_measurement = load_and_rearrange_data(
        uid, prefix="bba-measured", read_from_file=True,
        pv_for_applied_current="mux_power_converter_setpoint",
        pv_for_selected_magnet = "mux_selected_multiplexer_readback",
    )
    # preprocessed_measurement = load_and_rearrange_data_from_files(uid)
    # correct the bpm names
    if True:
        preprocessed_measurement = MeasurementData(
            measurement=[
                measurement_per_magnet_bpm_data_correct_name(m)
                for m in tqdm.tqdm(
                    preprocessed_measurement.measurement,
                    total=len(preprocessed_measurement.measurement),
                    desc="rename bpm raw   ",
                )
            ]
        )
    # convert bpm raw data to processed data
    # currently for debug to see where we are
    # meas = preprocessed_measurement.measurement
    preprocessed_measurement = MeasurementData(
        measurement=[measurement_per_magnet_bpms_raw_data_to_m(m, calib_repo=calib_repo) for m in
                     tqdm.tqdm(preprocessed_measurement.measurement, total=len(preprocessed_measurement.measurement),
                               desc="conv bpm raw -> m", )])
    # find out which elements were powered by the muxer
    # measurement / BESSY II epics environment uses upper case names
    # model uses lower case
    magnet_names = get_magnet_names(preprocessed_measurement)[:7]
    element_names_lc = [name.lower() for name in magnet_names]
    selected_model = load_model(required_element_names=element_names_lc, filename="bessyii_twiss_thor_scsi.nc"
                                # filename="bessyii_twiss_thor_scsi_twin_with_wavelength_shifter.nc",
                                )
    # control system uses upper case names ... fix the model here as long as required
    selected_model["pos"] = [name.upper() for name in selected_model.pos.values]

    # Display some info on the loaded model ...
    # not to be set off by how the phase advance is stored
    phase_advance = selected_model.mu
    logger.info(f"phase advance at end {phase_advance.isel(pos=-1)}")
    fit_ready_data = FitReadyData(
        per_magnet=[flatten_for_fit(measurement.per_magnet, measurement.name, pos="pos", rms="rms") for measurement in
                    tqdm.tqdm(preprocessed_measurement.measurement, desc="prepare fit data: ",
                              total=(len(preprocessed_measurement.measurement)))])

    t_theta = 1e-5  # 10 urad ... close to an average kick
    estimated_angles = EstimatedAngles(per_magnet=[  # can select pos="pos_raw" and rms="rms_raw"
        get_magnet_estimated_angle(fit_data, selected_model, t_theta, pos="pos", rms="rms") for fit_data in
        tqdm.tqdm(fit_ready_data.per_magnet, desc="est. equiv. angle: ", total=(len(fit_ready_data.per_magnet)), )],
        md="nothing", )
    # Connect to MongoDB
    # Convert the EstimatedAngles instance to a dictionary
    estimated_angles_dict = asdict(estimated_angles)
    fit_ready_data_dict = asdict(fit_ready_data)
    print("estimated angle calculated")
    # Create a InitializeMongo instance
    mongo_init = InitializeMongo()
    # Get the collection you need
    estimated_angles_collection = mongo_init.get_collection("estimatedangles")
    fit_read_data_collection = mongo_init.get_collection("fitreadydata")

    # estimated_angle_dict = estimated_angles
    estimated_angles_dict["uid"] = uid  # + "-shift"
    fit_ready_data_dict["uid"] = uid  # + "-shift"
    # Save the dictionary as a document in the collection
    estimated_angles_collection.insert_one(estimated_angles_dict)
    fit_read_data_collection.insert_one(fit_ready_data_dict)
    mongo_init.close_connection()


if __name__ == "__main__":
    import sys

    try:
        name, uid = sys.argv
    except ValueError:
        print("need one argument! a uid")
        if True:
            uid = "9ba454c7-f709-4c42-84b3-410b5ac05d9d"
            print(f"Using uid for testing  {uid}")
        else:
            sys.exit()
    main(uid)
