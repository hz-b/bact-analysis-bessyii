import logging
import os.path
from collections import OrderedDict as OrderedDictImpl

import numpy as np
import tqdm
import xarray as xr
from typing import Sequence

from pymongo import MongoClient

from bact_analysis.transverse.process import process_all_gen, combine_all
from bact_analysis.transverse.twiss_interpolate import interpolate_twiss
from bact_analysis.utils.preprocess import rename_doublicates, replace_names
from bact_analysis_bessyii.model.analysis_model import MeasurementData, MeasurementPerMagnet, DistortedOrbitUsedForKick, FitResult, MagnetEstimatedAngles, EstimatedAngles, EstimatedAngleForPlane
from bact_analysis_bessyii.model.analysis_util import flatten_for_fit
from bact_analysis_bessyii.model.calc import derive_angle
from bact_bessyii_ophyd.devices.pp.bpm_elem_util import rearrange_bpm_data
from bact_math_utils.distorted_orbit import closed_orbit_distortion
from .app_data import load_and_rearrange_data
from .calc import angles_to_offset_all

logger = logging.getLogger("bact-analysis-bessyii")

_m2mm = 1.0 / 1000.0

def process_rearranged_data(rearranged, bpm_names, bpm_names_as_in_model):
    da = rearrange_bpm_data(rearranged.bpm_elem_data)
    redm4proc = xr.merge(
        [
            da.sel(plane="x", quality="pos").rename("x_pos"),
            da.sel(plane="y", quality="pos").rename("y_pos"),
            da.sel(plane="x", quality="rms").rename("x_rms"),
            da.sel(plane="y", quality="rms").rename("y_rms"),
            rearranged.mux_power_converter_setpoint.rename("excitation"),
        ],
        compat='override'
    )
    # data is used further downstream for fitting
    # ensure data is processible by scipy.optimize.lstsq
    redm4proc = redm4proc.sel(bpm=bpm_names).rename_dims(bpm="pos").astype(float)

    return redm4proc


def magnet_data_to_common_names(
        rearranged: xr.Dataset, *, bpm_names, bpm_names_as_in_model, scale=_m2mm
) -> xr.Dataset:
    """Rename measurement variables to common names

    Selects data of known bpm names and renames them to the names
    of the model

    Todo:
        iterate over bpm data
        extract raw data for x and y taking scale and offset into accoungt
        calculate them to x and y in mm or m

    """
    return process_rearranged_data(rearranged, bpm_names, bpm_names_as_in_model)

def load_model(
        required_element_names: Sequence[str],
        filename: str = "bessyii_twiss_thor_scsi_from_twin.nc",
        datadir: str = None,
) -> Sequence[MeasurementData]:
    """
    """
    if datadir is None:
        datadir = os.path.dirname(__file__)

    path = os.path.join(datadir, filename)

    selected_model_ = xr.load_dataset(path)

    del selected_model_.coords["par"]

    doublets, renamed = rename_doublicates(
        selected_model_.coords["pos"].values.tolist()
    )
    if len(doublets):
        logger.warning(
            'Loaded model "%s": required to rename doubles: %s', path, doublets
        )
    selected_model_ = selected_model_.assign_coords(pos=renamed)

    # assume that the betatron change is small within these items ...
    # needs to be reviewed
    # Todo:
    #   element_dim is misleading: it means if data are from the start or the end of an element
    #   explain exactly what happens in the next 2 lines
    quad_twiss_ = interpolate_twiss(selected_model_, names=required_element_names)
    quad_twiss = (
        # for each selected quadrupole ... go back to positiom to select for the quadrupole in question
        quad_twiss_.rename(dict(elem="pos"))
        # .assign_coords(pos=quad_twiss_.coords["elem"].values)
        # .reset_coords(drop=True)
    )
    del quad_twiss_

    n_index = replace_names(
        list(selected_model_.coords["pos"].values),
        {name: name + "_s" for name in quad_twiss.coords["pos"].values},
    )
    selected_model = xr.concat(
        [selected_model_.assign_coords(pos=n_index), quad_twiss], dim="pos"
    ).sortby("ds")
    return selected_model

def get_magnet_names(preprocessed_measurement):
    # Initialize an empty list to store the names
    return [item.name for item in preprocessed_measurement.measurement if isinstance(item, MeasurementPerMagnet)]

def main(uid):
    preprocessed_measurement = load_and_rearrange_data(uid)

    # find out which elements were powered by the muxer
    # element_names = list(set(rearranged.mux_selected_multiplexer_readback.values.ravel()))
    # measurement / BESSY II epics environment uses upper case names
    # model uses lower case
    magnet_names = get_magnet_names(preprocessed_measurement)
    # element_names = [name for name in element_names if isinstance(name, str)]
    element_names_lc = [name.lower() for name in magnet_names]
    selected_model = load_model(required_element_names=element_names_lc)

    # Display some info on the loaded model ...
    # not to be set off by how the phase advance is stored
    phase_advance = selected_model.mu
    logger.info(f"phase advance at end {phase_advance.isel(pos=-1)}")
    # Beam position monitor names are core ... these are used to
    # match measurement data to model data.
    #
    # Currently the measured data uses bpm names as upper case
    # names the model lower case ones ...
    # Get the first MeasurementPerMagnet object
    first_magnet_measurement = preprocessed_measurement.measurement[0].per_magnet[0]

    # Get the BPM names from the first BPM object
    bpm_names = [bpm["name"] for bpm in first_magnet_measurement.bpm]
    bpm_names_lc = [name.lower() for name in bpm_names]

    # Reduced data set ... the sole data required for further
    # processing using standard names
    # here one could reduce the set to bpm's that are known to the
    # model and the measurement
    # reduced = magnet_data_to_common_names(
    #     preprocessed_measurement, bpm_names=bpm_names, bpm_names_as_in_model=bpm_names_lc
    # )

    # Estimate the angles an equivalent kicker would make
    # element_names = reduced.coords["name"].values
    #
    # orbit = DistortedOrbitUsedForKick(
    #     kick_strength=0.5,  # Replace with the actual kick strength
    #     # delta=...,  # Replace with the actual delta data
    # )
    #
    # # Create an instance of FitResult
    # equivalent_angle = FitResult(
    #     value=0.25,
    #     std=0.01,
    # )
    #
    # # Create an instance of FitResult for bpm offsets
    # bpm_offsets = [
    #     FitResult(value=0.05, std=0.002),
    #     FitResult(value=0.03, std=0.001),
    # ]
    #
    # # Create an instance of FitResult for the derived offset
    # offset = FitResult(
    #     value=0.12,
    #     std=0.005,
    # )
    #
    # # Create an instance of EstimatedAngleForPlane for the 'x' plane
    # estimated_angle_x = EstimatedAngleForPlane(
    #     orbit=orbit,
    #     equivalent_angle=equivalent_angle,
    #     bpm_offsets=bpm_offsets,
    #     offset=offset,
    # )
    #
    # # Create an instance of EstimatedAngleForPlane for the 'y' plane
    # estimated_angle_y = EstimatedAngleForPlane(
    #     orbit=orbit,
    #     equivalent_angle=equivalent_angle,
    #     bpm_offsets=bpm_offsets,
    #     offset=offset,
    # )
    #
    # # Create an instance of EstimatedAngle
    # estimated_angle = EstimatedAngle(
    #     name="Some Magnet Name",
    #     x=estimated_angle_x,
    #     y=estimated_angle_y,
    # )
    #
    #
    #



    # print("element_names", element_names)
    # Todo:
    #  test with weights when taking data with rms it is working
    #  e.g . from real measurement data
    #     beta: np.ndarray,
    #     mu: np.ndarray,
    #     *,
    #     tune: float,
    #     beta_i: float,
    #     theta_i: float,
    #     mu_i: float
    t_theta = 1e-5 # 10 urad ... close to an average kick
    distorted_orbit_x = closed_orbit_distortion(
        selected_model.beta.sel(plane="x").values, selected_model.mu.sel(plane="x").values * 2 * np.pi,
        tune = selected_model.mu.sel(plane="x").values[-1],
        beta_i=selected_model.beta.sel(plane="x", pos="q1m1t1r").values,
        mu_i = selected_model.mu.sel(plane="x", pos="q1m1t1r").values * 2 * np.pi,
        theta_i = t_theta,
    )
    # one magnet one plane
    kick_x = DistortedOrbitUsedForKick(kick_strength=t_theta, delta=OrderedDictImpl(zip(selected_model.coords['pos'].values, distorted_orbit_x)))
    flattened = flatten_for_fit(preprocessed_measurement.measurement[0])
    # flatten_data_as_list = get_data_as_lists(flatten.x)

    angle_x = derive_angle(kick_x, flattened.x, flattened.excitations)
    # angle_y = derive_angle(kick_y, flattened.y, flattened.excitations)

    #
    # ds_elems = (
    #     selected_model.ds.sel(pos=element_names_lc)
    #     .rename(pos="name")
    #     .assign_coords(name=element_names)
    # )
    # estimated_angles = (
    #     combine_all(estimated_angles_)
    #     .merge(dict(ds=selected_model.ds, ds_elems=ds_elems))
    #     .sortby(["ds_elems"])
    # )
    # preprocessed_measurement_data = reduced.swap_dims({'pos' : 'bpm'})
    # # preprocessed_measurement_data.name = "preprocessed_measurement_data"
    # # preprocessed_measurement_data = reduced.rename(pos="bpm_pos")
    #
    #
    # # estimated_angles = estimated_angles.merge(preprocessed_measurement_data)
    #
    # offsets = angles_to_offset_all(
    #     estimated_angles, names=element_names, tf_scale=75.0 / 28.0
    # )

    preprocessed_measurement_data.to_netcdf(f"preprocessed_measurement_data_{uid}.nc")
    estimated_angles.to_netcdf(f"estimated_angles_{uid}.nc")
    offsets.to_netcdf(f"offsets_{uid}.nc")


    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
    db = client["bessyii"]  # Replace "mydatabase" with your desired database name
    offset_collection = db["offsets"]  
    estimated_angles_collection = db["estimatedangles"]  
    preprocessed_measurement_collection = db["preprocessedmeasurement"]  

    estimated_angle_dict = estimated_angles.to_dict()
    estimated_angle_dict['uid'] = uid
    # Save the dictionary as a document in the collection
    estimated_angles_collection.insert_one(estimated_angle_dict)

    offsets_dict = offsets.to_dict()
    offsets_dict['uid'] = uid
    # Save the dictionary as a document in the collection
    offset_collection.insert_one(offsets_dict)

    preprocessed_measurement_dict = preprocessed_measurement_data.to_dict()
    preprocessed_measurement_dict['uid'] = uid
    # Save the dictionary as a document in the collection
    preprocessed_measurement_collection.insert_one(preprocessed_measurement_dict)
    client.close()

if __name__ == "__main__":
    import sys

    try:
        uid, = sys.argv
    except ValueError:
        print("need one argument! a uid")
        if True:
            uid = '9ba454c7-f709-4c42-84b3-410b5ac05d9d'
            print(f"Using uid for testing {uid}")
        else:
            sys.exit()
    main(sys.argv[0])
