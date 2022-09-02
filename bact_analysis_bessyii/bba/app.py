from .app_data import load_and_rearrange_data
from .calc import angles_to_offset_all
from bact_analysis.utils.preprocess import rename_doublicates, replace_names
from bact_analysis.transverse.process import process_all_gen, combine_all
from bact_analysis.transverse.twiss_interpolate import interpolate_twiss
import xarray as xr
import tqdm
from typing import Sequence
import logging
import os.path

logger = logging.getLogger("bact-analysis-bessyii")


_m2mm = 1.0 / 1000.0


def magnet_data_to_common_names(
    rearranged: xr.Dataset, *, bpm_names, bpm_names_as_in_model, scale=_m2mm
) -> xr.Dataset:
    """Rename measurement variables to common names

    Selects data of known bpm names and renames them to the names
    of the model
    """
    measurement_vars = dict(
        dt_bpm_waveform_x_pos="x_pos",
        dt_bpm_waveform_y_pos="y_pos",
        dt_bpm_waveform_x_rms="x_rms",
        dt_bpm_waveform_y_rms="y_rms",
        dt_mux_power_converter_setpoint="excitation",
    )
    redm4proc = (
        rearranged[list(measurement_vars.keys())]
        .rename_vars(**measurement_vars)
        .sel(bpm=bpm_names)
        .rename_dims(bpm="pos")
        .assign_coords(pos=bpm_names_as_in_model)
        .reset_coords(drop=True)
    )
    # BPM Data are in mm
    redm4proc["x_pos"] = redm4proc.x_pos * scale
    redm4proc["y_pos"] = redm4proc.y_pos * scale
    redm4proc["x_rms"] = redm4proc.x_rms * scale
    redm4proc["y_rms"] = redm4proc.y_rms * scale
    return redm4proc


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
    quad_twiss_ = interpolate_twiss(selected_model_, names=required_element_names)
    quad_twiss = (
        quad_twiss_.rename_dims(name="pos")
        .assign_coords(pos=quad_twiss_.coords["name"].values)
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

    element_names = list(set(rearranged.dt_mux_selector_selected.values.ravel()))
    # measurement / BESSY II epics environment uses upper case names
    # model uses lower case
    element_names_lc = [name.lower() for name in element_names]
    selected_model = load_model( required_element_names=element_names_lc)
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
    estimated_angles_ = {
        name: item
        for name, item in tqdm.tqdm(
            process_all_gen(
                selected_model,
                reduced,
                element_names,
                bpm_names=bpm_names_lc,
                theta=1e-5,
                use_weights=True,
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
        sys.exit()
    main(sys.argv[0])
