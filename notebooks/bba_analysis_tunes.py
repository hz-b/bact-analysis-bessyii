#!/usr/bin/env python


import datetime
from dataclasses import dataclass
import functools
import tqdm

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np

import bact_analysis.utils.preprocess
import bact_analysis_bessyii.bba.preprocess_data
from bact_analysis.transverse.twiss_interpolate import (
    data_for_elements,
    interpolate_twiss,
)

from bact_math_utils.linear_fit import linear_fit_1d
from bba_data import db
import logging
from enum import IntEnum

logger = logging.getLogger("bact-analysis-bessyii")

uids = [
    # First run in with power converters
    "1e6ec7f3-44a5-4e5c-a798-1d7ce12aafe1",
    # repeated runs
    "22354cec-864b-4f38-ad46-a9641d07d1ac",
    "eb89753c-5388-4ebb-a992-fc25b478acd8",
    "b226943c-1941-46ce-bc35-0530ea6e276c",
    "e0aef7b8-f57e-4594-9618-70d01aaa65a7",
    "e60215ff-62ea-4d3b-a968-f6b0d9d9ee9d",
    "fa22af2e-0398-41eb-94b9-e9b957ba4f31",
]

figs = dict()

quad_length = dict(
    Q4M1=0.5333,
    Q4M2=0.5329,
    Q1M1=0.2856,
    Q1M2=0.2854,
    Q3M2=0.2875,
    Q3M1=0.2923,
    Q2M1=0.2326,
    Q2M2=0.2326,
    Q5M=0.2325,
)

quad_length = dict(
    Q4=0.5,
    Q1=0.25,
    Q2=0.2,
    Q3=0.25,
    Q5=0.2,
    )

@dataclass
class FigAxes:
    fig: object
    axes: object
    twin_axes: object


# @functools.lru_cache(maxsize=None)
def create_fig(*, num, nrows=2, ncols=1, twin_axes=False):
    r = None
    try:
        r = figs[num]
    except KeyError:
        pass
    if r is not None:
        return r

    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, num=num, clear=False, figsize=[16, 6]
    )
    ta = None
    if twin_axes:
        ta = [plt.twinx(ax) for ax in axes]
    tmp = FigAxes(fig=fig, axes=axes, twin_axes=ta)
    figs[num] = tmp
    return figs[num]


@dataclass
class TuneFitResult:
    x: np.ndarray
    std: np.ndarray


class Polarity(IntEnum):
    pos = 1
    neg = -1


@dataclass
class MuxerPolarity:
    x: Polarity
    y: Polarity


muxer_polarities = dict(
    # Q1=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
    # Q2=MuxerPolarity(x=Polarity.neg, y=Polarity.pos),
    # Q3=MuxerPolarity(x=Polarity.neg, y=Polarity.pos),
    # Q4=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
    # Q5=MuxerPolarity(x=Polarity.neg, y=Polarity.pos),
    Q1=MuxerPolarity(x=Polarity.neg, y=Polarity.pos),
    Q2=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
    Q3=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
    Q4=MuxerPolarity(x=Polarity.neg, y=Polarity.pos),
    Q5=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
    Q1M1T=MuxerPolarity(x=Polarity.neg, y=Polarity.pos),
    Q2M1T=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
    Q3M1T=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
    Q4M1T=MuxerPolarity(x=Polarity.neg, y=Polarity.pos),
    Q5M1T=MuxerPolarity(x=Polarity.pos, y=Polarity.neg),
)


def fit_tune_shift(dI, tune):
    x, std = linear_fit_1d(dI, tune)
    return TuneFitResult(x=x, std=std)


def fit_tune_shift_all(ds, name):
    x = fit_tune_shift(ds.excitation, ds.x_tune)
    y = fit_tune_shift(ds.excitation, ds.y_tune)

    ndx = xr.DataArray(
        name="x",
        data=[[x.x, x.std]],
        dims=["name", "res", "coeff"],
        coords=[[name], ["val", "std"], ["slope", "intercept"]],
    )

    ndy = xr.DataArray(
        name="y",
        data=[[y.x, y.std]],
        dims=["name", "res", "coeff"],
        coords=[[name], ["val", "std"], ["slope", "intercept"]],
    )

    nds = xr.merge([ndx, ndy])
    return nds


def calculate_dn_from_beta(*, beta_s, beta_e, L):
    dB = beta_e - beta_s
    scale = L / dB
    dev_start, dev_end = [np.log(beta * L) for beta in (beta_s, beta_e)]
    dev = dev_end - dev_start
    res = scale * dev
    res = res / (2 * np.pi)
    return res


def phase_advance_dbeta(ds, names):
    res = data_for_elements(ds, names)
    dbeta = res.beta.sel(element="end") - res.beta.sel(element="start")
    dmu = res.mu.sel(element="end") - res.mu.sel(element="start")
    L = res.ds.sel(element="end") - res.ds.sel(element="start")
    dmu_from_beta = calculate_dn_from_beta(
        beta_s=res.beta.sel(element="start"), beta_e=res.beta.sel(element="end"), L=L
    )
    return res.update(dict(dmu=dmu, L=L, dmu_from_beta=dmu_from_beta, dbeta=dbeta))


@functools.lru_cache(maxsize=1)
def load_model():
    return xr.load_dataset("bessyii_twiss_thor_scsi.nc")
    # return xr.load_dataset("bessii_twiss_tracy.nc")

@functools.lru_cache(maxsize=1)
def load_quad_k():
    df =  pd.read_json("bessy2_quad_strength_measured_loco.json")
    df.loc[:, "name"] = [name.lower() for name in df.loc[:, "name"]]
    return df.set_index("name")

def main(uid, color="b", marker="."):

    run = db[uid]

    (descriptor,) = run.primary.metadata["descriptors"]
    configuration = descriptor["configuration"]
    dt_configuration = configuration["dt"]
    all_data_ = run.primary.to_dask()

    for name, item in tqdm.tqdm(all_data_.items(), total=len(all_data_.variables)):
        item.load()

    muxer_pc_current_change = bact_analysis.utils.preprocess.enumerate_changed_value(
        all_data_.dt_mux_power_converter_setpoint
    )
    muxer_pc_current_change.name = "muxer_pc_current_change"
    muxer_or_pc_current_change = (
        bact_analysis.utils.preprocess.enumerate_changed_value_pairs(
            all_data_.dt_mux_power_converter_setpoint,
            all_data_.dt_mux_selector_selected,
        )
    )
    muxer_or_pc_current_change.name = "muxer_or_pc_current_change"
    bpm_names = all_data_.dt_bpm_waveform_names.values[0]

    bpm_dims = bact_analysis_bessyii.bba.preprocess_data.replaceable_dims_bpm(
        all_data_, prefix="dt_", expected_length=len(bpm_names)
    )

    replace_dims = {dim: "bpm" for dim in bpm_dims}
    # replace_dims.update({dim : 'pos' for dim in beam_dims})
    all_data = all_data_.rename(replace_dims).assign_coords(bpm=list(bpm_names))

    preprocessed = xr.merge(
        [all_data, muxer_pc_current_change, muxer_or_pc_current_change]
    )

    rearranged = xr.concat(
        bact_analysis.utils.preprocess.reorder_by_groups(
            preprocessed,
            preprocessed.groupby(preprocessed.dt_mux_selector_selected),
            reordered_dim="name",
            dim_sel="time",
            new_indices_dim="step",
        ),
        dim="name",
    )

    tune_data = xr.merge(
        [
            rearranged.dt_mr_tune_fb_hor_readback,
            rearranged.dt_mr_tune_fb_vert_readback,
            rearranged.dt_mux_power_converter_setpoint,
            rearranged.dt_mux_power_converter_readback,
        ]
    )

    model_data = load_model()
    quad_k_data = load_quad_k()

    measurement_vars = dict(
        dt_bpm_waveform_x_pos="x_pos",
        dt_bpm_waveform_y_pos="y_pos",
        dt_mr_tune_fb_hor_readback="x_tune",
        dt_mr_tune_fb_vert_readback="y_tune",
        dt_mux_power_converter_setpoint="excitation",
    )

    redm4proc = (
        rearranged[list(measurement_vars.keys())]
        .rename_vars(**measurement_vars)
        .sel(bpm=bpm_names)
    )

    tune_fits = xr.concat(
        [
            fit_tune_shift_all(redm4proc.sel(name=name), name)
            for name in redm4proc.coords["name"].values
        ],
        dim="name",
    )

    def adjust_tick(tick):
        tick.set_fontsize("small")
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
        tick.set_verticalalignment("top")

    # plot measured tunes
    magnet_families = ["Q4", "Q2", "Q3", "Q5", "Q1"]
    # magnet_families = ["Q4M1T", "Q2M1T", "Q3M1T", "Q5M1T", "Q1M1T"]
    for fignum, family_name in enumerate(magnet_families):
        sel = tune_fits.isel(
            name=[
                name[: len(family_name)] == family_name
                for name in tune_fits.coords["name"].values
            ]
        )

        pol = muxer_polarities[family_name]
        # a consistency issue ..
        names = [name.lower() for name in sel.coords["name"].values]
        logger.debug("Interpolating twiss for %s", names)

        # t_twiss = phase_advance_dbeta(model_data, names)
        t_twiss = interpolate_twiss(model_data, names)

        # check that order is correct
        not_matching_names = [
            (nsel, ntwiss)
            for ntwiss, nsel in zip(
                t_twiss.coords["name"].values, sel.coords["name"].values
            )
            if nsel.lower() != ntwiss
        ]
        if not_matching_names:
            logger.error("Name sequence does not match for : %s", not_matching_names)
        else:
            sel.coords["name"] = [name.lower() for name in sel.coords["name"].values]
        # inconsistent naming ... hide the names for one of them
        beta_x = t_twiss.beta.sel(plane="x")  # .values
        beta_y = t_twiss.beta.sel(plane="y")  # .values

        q_l = quad_length[family_name]
        s_x = beta_x * pol.x
        s_y = beta_y * pol.y

        # from Peter's hand note
        delta_g = 0.796 / 5.
        brho = 5.67044
        delta_k = delta_g / brho
        pinv4 = 1 / (np.pi * 4)
        dq_x = beta_x * delta_k * q_l * pol.x * pinv4
        dq_y = beta_y * delta_k * q_l * pol.y * pinv4

        f_rev = 1.25e6
        df_x = dq_x  * f_rev / 1e3
        df_y = dq_y  * f_rev / 1e3
        # --------------------------------------------------------
        # plot the tunes as measured
        t_fig = create_fig(num=fignum + 100, nrows=3, ncols=1, twin_axes=True)

        # Same axes every run
        ax_x, ax_y, ax_beta = t_fig.axes
        ta_x, ta_y, ta_beta = t_fig.twin_axes
        line, x_err, y_err = ax_x.errorbar(
            names,
            sel.x.sel(res="val", coeff="slope"),
            yerr=sel.x.sel(res="std", coeff="slope"),
            linestyle="",
            marker=marker,
            color=color,
        )
        ax_x.plot(names, df_x, "x--", linewidth=0.5, color=color)
        #ta_x.plot(names, s_x, linestyle="-", color=color, linewidth=0.5, marker=".")
        pol_txt = "- " if pol.x == Polarity.neg else ""
        ta_x.set_ylabel(pol_txt + r"$\beta_x$ [m]")

        ax_x.set_ylabel("dQ$_x$/dI [kHz/A]")
        ax_y.errorbar(
            names,
            sel.y.sel(res="val", coeff="slope"),
            yerr=sel.y.sel(res="std", coeff="slope"),
            linestyle="",
            marker=marker,
            color=line.get_color(),
        )
        ax_y.plot(names, df_y, "x--", linewidth=0.5, color=color)
        #ta_y.plot(names, s_y, linestyle="-", color=color, linewidth=0.5, marker=".")
        pol_txt = "- " if pol.y == Polarity.neg else ""
        ta_y.set_ylabel(pol_txt + r"$\beta_y$ [m]")

        ax_y.set_ylabel("dQ$_y$/dI [kHz/A]")
        ax_beta.plot(names, beta_x, linestyle="-", linewidth=0.5, marker=marker)
        ax_beta.plot(names, beta_y, linestyle="-.", linewidth=0.5, marker=marker)
        ax_beta.set_ylabel(r"$mu_x, \mu_y$ [m]")

        [adjust_tick(tick) for tick in ax_beta.get_xmajorticklabels()]
        continue
        # --------------------------------------------------------
        # plot the tunes as measured but scaled by betatron function
        t_fig = create_fig(num=fignum + 100 + 20)

        # Same axes every run
        ax_x, ax_y = t_fig.axes

        line, x_err, y_err = ax_x.errorbar(
            names,
            sel.x.sel(res="val", coeff="slope") / (dq_x),
            yerr=sel.x.sel(res="std", coeff="slope") / (dq_x),
            linestyle="",
            marker=marker,
            color=color,
        )
        ax_x.set_ylabel(r"$dQ_x/dI\beta_x$ [kHz/A m]")

        ax_y.errorbar(
            names,
            sel.y.sel(res="val", coeff="slope") / (dq_y),
            yerr=sel.y.sel(res="std", coeff="slope") / (dq_y),
            linestyle="",
            marker=marker,
            color=line.get_color(),
        )
        ax_y.set_ylabel(r"$dQ_x/dI\beta_y$ [kHz/A m]")
        [adjust_tick(tick) for tick in ax_y.get_xmajorticklabels()]


def main_all():
    """Follow formatting of first elog enty

    See elog http://elog-v2.trs.bessy.de:8080/Machine+Devel.,+Comm./1976

        1 	x 	b 	  	1e6ec7f3-44a5-4e5c-a798-1d7ce12aafe1
        2 	+ 	red 	  	22354cec-864b-4f38-ad46-a9641d07d1ac
        3 	+ 	green 	  	eb89753c-5388-4ebb-a992-fc25b478acd8
        1 	^ 	cyan 	  	b226943c-1941-46ce-bc35-0530ea6e276c
        2 	^ 	magenta 	  	e0aef7b8-f57e-4594-9618-70d01aaa65a7
        3 	v 	cyan 	  	e60215ff-62ea-4d3b-a968-f6b0d9d9ee9d
        4 	v 	green 	  	fa22af2e-0398-41eb-94b9-e9b957ba4f31

    """
    colors = "b", "r", "g", "c", "m", "c", "g"
    markers = "x", "+", "+", "^", "^", "v", "v"

    for uid, c, m in zip(uids, colors, markers):
        main(uid, color=c, marker=m)

    quad_family = dict(Q1=104, Q5=103, Q2=101, Q4=100, Q3=102)
    for fam_name, num in quad_family.items():
        figname = f"{fam_name}_tune_data"
        fig = plt.figure(num)
        fig.savefig(figname + ".pdf")
        fig.savefig(figname + ".png", dpi=600)
