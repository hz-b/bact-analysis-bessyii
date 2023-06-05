import logging
import os.path

import numpy as np
import tqdm
import xarray as xr
from typing import Sequence, Hashable

from bact_analysis.transverse.process import process_all_gen, combine_all
from bact_analysis.transverse.twiss_interpolate import interpolate_twiss
from bact_analysis.utils.preprocess import rename_doublicates, replace_names
from .app_data import load_and_rearrange_data
from .calc import angles_to_offset_all

logger = logging.getLogger("bact-analysis-bessyii")

_m2mm = 1.0 / 1000.0


# def bpm_to_dataset(read_data: Sequence[Hashable]) -> xr.DataArray:
#     """
#
#     todo:
#         how to include calibration data here?
#     """
#
#     def extract_data(item):
#         try:
#             r = (
#                 [item["x"]["pos_raw"], item["x"]["rms_raw"]],
#                 [item["y"]["pos_raw"], item["y"]["rms_raw"]]
#             )
#         except KeyError as ex:
#             logger.error(f"Failed to treat item {item}: {ex}")
#             raise ex
#         return r
#
#     d = {item['name']: extract_data(item) for item in read_data}
#     data = [item for _, item in d.items()]
#     bpm_names = list(d.keys())
#     try:
#         da = xr.DataArray(data=data, dims=["bpm", "plane", "quality"],
#                           coords=[bpm_names, ["x", "y"], ["pos", "rms"]])
#     except Exception as ex:
#         logger.error(f"Failed to convert dic to xarray {ex}")
#         logger.error(f"dict was {d}")
#         raise ex
#     return da
def extract_data(item):
    try:
        x_pos_raw = item["x"]["pos_raw"]
        x_rms_raw = item["x"]["rms_raw"]
        y_pos_raw = item["y"]["pos_raw"]
        y_rms_raw = item["y"]["rms_raw"]
    except KeyError as ex:
        logger.error(f"Failed to treat item {item}: {ex}")
        raise ex
    return ([x_pos_raw, x_rms_raw], [y_pos_raw, y_rms_raw])


def bpm_to_dataset(read_data: Sequence[Hashable]) -> xr.DataArray:
    """
    Convert BPM data to an xarray DataArray.

    Parameters:
        read_data (Sequence[Hashable]): List of BPM data dictionaries.

    Returns:
        xr.DataArray: Converted xarray DataArray.

    Raises:
        KeyError: If any required keys are missing in the data dictionaries.
        Exception: If there is an error converting the data to an xarray DataArray.
    """

    d = {item['name']: extract_data(item) for item in read_data}
    data = [item for _, item in d.items()]
    bpm_names = list(d.keys())
    try:
        da = xr.DataArray(data=data, dims=["bpm", "plane", "quality"],
                          coords=[bpm_names, ["x", "y"], ["pos", "rms"]])
    except Exception as ex:
        logger.error(f"Failed to convert dict to xarray: {ex}")
        logger.error(f"Dict was: {d}")
        raise ex
    return da


def rearrange_bpm_data(rearranged):
    data = rearranged.bpm_elem_data
    r = [[bpm_to_dataset(data.isel(name=name_idx, step=step_idx).values)
          for step_idx in range(len(data.coords["step"]))]
         for name_idx in range(len(data.coords["name"]))]
    ref_item = r[0][0]
    bpm_names_as_in_model = data.coords["bpm"].values
    dims = ["name", "step"] + list(ref_item.dims)
    shape = (len(data.coords["name"]), len(data.coords["step"])) + ref_item.shape
    da = xr.DataArray(np.empty(shape, dtype=object), dims=dims)
    for name_idx in range(len(data.coords["name"])):
        for step_idx in range(len(data.coords["step"])):
            da[name_idx, step_idx] = r[name_idx][step_idx]
    da = da.assign_coords(dict(
        name=data.coords["name"],
        step=data.coords["step"],
        bpm=ref_item.coords["bpm"],
        plane=ref_item.coords["plane"],
        quality=ref_item.coords["quality"]
    ))
    return da


def process_rearranged_data(rearranged, bpm_names, bpm_names_as_in_model):
    da = rearrange_bpm_data(rearranged)
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
    redm4proc = redm4proc.sel(bpm=bpm_names).rename_dims(bpm="pos")

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

    # commented a working version until ......
    # tmp = rearranged.bpm_elem_data.isel(name=0, step=0).values
    #
    # data = rearranged.bpm_elem_data
    # r = []
    # for name_idx in range(len(data.coords["name"])):
    #     l = []
    #     for step_idx in range(len(data.coords["step"])):
    #         tmp = rearranged.bpm_elem_data.isel(name=name_idx, step=step_idx).values
    #         l.append(bpm_to_dataset(tmp))
    #     r.append(l)
    # ref_item = r[0][0]
    # dims = ["name", "step"] + list(ref_item.dims)
    # # xr.combine_nested()
    # da = xr.DataArray(r, dims=dims)
    # da = da.assign_coords(dict(name=data.coords["name"], step=data.coords["step"], bpm=ref_item.coords["bpm"],
    #                            plane=ref_item.coords["plane"], quality=ref_item.coords["quality"]))
    # da


#    ............
# ds = xr.DataArray(r)
# [bpm_to_dataset(v) for v in ]
# measurement_vars = dict(
#     dt_bpm_waveform_x_pos="x_pos",
#     dt_bpm_waveform_y_pos="y_pos",
#     dt_bpm_waveform_x_rms="x_rms",
#     dt_bpm_waveform_y_rms="y_rms",
#     dt_mux_power_converter_setpoint="excitation",
# )

# todo:
# apply conversion factors (scaling is enough!)
# .......................
# redm4proc = xr.merge(
#     dict(
#         x_pos=da.sel(plane="x", quality="pos"),
#         y_pos=da.sel(plane="y", quality="pos"),
#         x_rms=da.sel(plane="x", quality="rms"),
#         y_rms=da.sel(plane="y", quality="rms"),
#         excitation=rearranged.mux_power_converter_setpoint
#     )
# )
# redm4proc = redm4proc.sel(bpm=bpm_names).rename_dims(bpm="pos").assign_coords(pos=bpm_names_as_in_model)

# ............

#     .reset_coords(drop=True)
# redm4proc = (
#     rearranged[list(measurement_vars.keys())]
#     .rename_vars(**measurement_vars)
#     .sel(bpm=bpm_names)
#
#
# )
# # BPM Data are in mm
# redm4proc["x_pos"] = redm4proc.x_pos * scale
# redm4proc["y_pos"] = redm4proc.y_pos * scale
# redm4proc["x_rms"] = redm4proc.x_rms * scale
# redm4proc["y_rms"] = redm4proc.y_rms * scale
#
# return redm4proc


def load_model(
        required_element_names: Sequence[str],
        filename: str = "bessyii_twiss_thor_scsi.nc",
        datadir: str = None,
):
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
        # for each selected quadrupole ... go back to
        quad_twiss_.rename(dict(elem="pos"))
        .assign_coords(pos=quad_twiss_.coords["elem"].values)
        .reset_coords(drop=True)
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


def main(uid):
    rearranged, dt_configuration = load_and_rearrange_data(uid)

    # find out which elements were powered by the muxer
    element_names = list(set(rearranged.mux_selected_multiplexer_readback.values.ravel()))
    # measurement / BESSY II epics environment uses upper case names
    # model uses lower case
    element_names = [name for name in element_names if isinstance(name, str)]
    element_names_lc = [name.lower() for name in element_names]
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
    bpm_names = rearranged.coords["bpm"]
    bpm_names_lc = [name.lower() for name in bpm_names.values]

    # Reduced data set ... the sole data required for further
    # processing using standard names
    # here one could reduce the set to bpm's that are known to the
    # model and the measurement
    reduced = magnet_data_to_common_names(
        rearranged, bpm_names=bpm_names, bpm_names_as_in_model=bpm_names_lc
    )

    # Estimate the angles an equivalent kicker would make
    element_names = reduced.coords["name"].values
    # print("element_names", element_names)
    # Todo:
    #  test with weights when taking data with rms it is working
    #  e.g . from real measurement data
    #
    estimated_angles_ = {
        name: item
        for name, item in tqdm.tqdm(
            process_all_gen(
                selected_model,
                reduced,
                element_names,
                bpm_names=bpm_names_lc,
                theta=1e-5,
                use_weights=False,
            ),
            total=len(element_names),
            desc="calculating equivalent kick angles",
        )
    }

    ds_elems = (
        selected_model.ds.sel(pos=element_names_lc)
        .rename(pos="name")
        .assign_coords(name=element_names)
    )
    estimated_angles = (
        combine_all(estimated_angles_)
        .merge(dict(ds=selected_model.ds, ds_elems=ds_elems))
        .sortby(["ds_elems"])
    )
    preprocessed_measurement_data = reduced.swap_dims({'pos' : 'bpm'})
    # preprocessed_measurement_data.name = "preprocessed_measurement_data"
    # preprocessed_measurement_data = reduced.rename(pos="bpm_pos")


    estimated_angles = estimated_angles.merge(preprocessed_measurement_data)
    # estimated_angles = xr.merge([estimated_angles, preprocessed_measurement_data])

    offsets = angles_to_offset_all(
        estimated_angles, names=element_names, tf_scale=75.0 / 28.0
    )

    estimated_angles.to_netcdf(f"estimated_angles_{uid}.nc")
    offsets.to_netcdf(f"offsets_{uid}.nc")


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
