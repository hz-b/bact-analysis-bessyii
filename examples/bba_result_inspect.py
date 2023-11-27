import logging

from bact_bessyii_mls_ophyd.db.mongo_repository import InitializeMongo
from pymongo import MongoClient
import jsons
import numpy as np
import pandas as pd
from bact_analysis_bessyii.model.analysis_model import (
    ErrorEstimates,
    EstimatedAngles,
    MagnetEstimatedAngles,
)
from bact_math_utils.stats import mean_absolute_error
import logging

logger = logging.getLogger("bact-bessyii-analysis")


def post_process(error_estimates: ErrorEstimates):
    return (
        mean_absolute_error(error_estimates.mean_absolute_error),
        # np.sqrt(mean_square_error(np.sqrt(error_estimates.mean_square_error))) * pscale
        np.sqrt(np.sum(error_estimates.mean_square_error)),
    )


def overview_dataframe(estimated_angles: ErrorEstimates) -> pd.DataFrame:
    data = {
        est_for_mag.name: (
            est_for_mag.x.offset.value,
            est_for_mag.x.offset.std,
            est_for_mag.y.offset.value,
            est_for_mag.y.offset.std,
        )
        + post_process(est_for_mag.x.error_estimates)
        + post_process(est_for_mag.y.error_estimates)
        + (0e0,)
        for est_for_mag in estimated_angles.per_magnet
    }

    df = pd.DataFrame(
        data,
        index=["x_o", "x_std", "y_o", "y_std", "x_mae", "x_mse", "y_mae", "y_mse", "s"],
    ).T
    return df


def plot_overview_dataframe(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    # scale data from mm to m
    pscale = np.ones(df.shape[1]) * 1e3
    # error estimates are more or less squared
    pscale[4:] = pscale[4:] ** 2
    df = df * pscale
    fig, axes = plt.subplots(4, 1, sharex=True)
    ax, ax_mse, ax_mae, ax_ratio = axes

    ax.errorbar(df.index, df.x_o, yerr=df.x_std, fmt="+-", label="x o")
    ax.errorbar(df.index, df.y_o, yerr=df.y_std, fmt="x--", label="y o")
    ax.set_ylabel("x,y [mm]")

    ax_mse.plot(df.x_mse, "+-", label="x mse")
    ax_mse.plot(df.y_mse, "x--", label="y mse")
    ax_mse.set_ylabel(r"$\Delta$x, $\Delta$y [um]")

    ax_mae.plot(df.x_mae, "+-", label="x mae")
    ax_mae.plot(df.y_mae, "x-.", label="y mae")
    ax_mae.set_ylabel(r"$\Delta$x, $\Delta$y [um]")

    ax_ratio.plot(df.x_mse / df.x_mae, "+-", label="x: mse/mae"),
    ax_ratio.plot(df.y_mse / df.y_mae, "x-", label="y: mse/mae"),
    ax_ratio.set_ylabel(r"mse / mae")
    ax.legend()
    ax_mse.legend()
    ax_mae.legend()
    ax_ratio.legend()
    plt.setp(ax_ratio.xaxis.get_ticklabels(), "verticalalignment", "top")
    plt.setp(ax_ratio.xaxis.get_ticklabels(), "horizontalalignment", "right")
    plt.setp(ax_ratio.xaxis.get_ticklabels(), "rotation", 45)
    del axes, ax, ax_mse, ax_mae, ax_ratio

    # And now the hack to add peter's data
    dfp = pd.read_csv("23102501.ERG", delimiter=",", header=0)
    dfp.columns = ["index", "quad_name", "s", "x_o", "x_std", "y_o", "y_std"]
    dfp.quad_name = [name.strip() for name in dfp.quad_name]
    dfp = dfp.set_index("quad_name", drop=True)

    fig, axes = plt.subplots(2, 1, sharex=True)
    ax, ax_diff = axes
    df = df.loc[dfp.index, :]
    # copy s position
    for name in dfp.index:
        df.loc[name, "s"] = dfp.loc[name, "s"]
    df = df.sort_values(by="s")

    for ref, chk in zip(df.index, dfp.index):
        if ref != chk:
            logger.warning(f"quad names do not match {ref} != {chk}")

    ax.plot(dfp.s, "-", df.s, "-")
    ax_diff.plot(dfp.s.index, df.s.values - dfp.s.values)
    ax_diff.set_xticks(np.arange(len(dfp.index)))
    ax_diff.set_xticklabels(dfp.index)
    plt.setp(
        ax_diff.xaxis.get_ticklabels(),
        "horizontalalignment",
        "right",
        "verticalalignment",
        "top",
        "rotation",
        45,
    )

    del axes, ax, ax_diff

    cooking_factor = 1 / 2.0

    fig, axes = plt.subplots(3, 1, sharex=True)
    ax_x, ax_y, ax_ratio = axes
    # fmt: off
    ax_x.errorbar(dfp.s, dfp.x_o, yerr=dfp.x_std, fmt="x-", label="$x_p$")
    ax_y.errorbar(dfp.s, dfp.y_o, yerr=dfp.y_std, fmt="x-", label="$y_p$")
    ax_y.set_xticks(dfp.s)
    ax_y.set_xticklabels(dfp.index)
    plt.setp(ax_y.get_xticklabels(), "horizontalalignment", "right", "verticalalignment", "top","rotation", 45)
    # plt.setp(ax_y.get_xticklabels(), )
    # plt.setp(ax_y.get_xticklabels(), )

    ax_x.errorbar(df.s, -df.x_o * cooking_factor, yerr=df.x_std, fmt="+-", label="$x_o$")
    ax_y.errorbar(df.s,  df.y_o * cooking_factor, yerr=df.y_std, fmt="+-", label="$y_o$")
    # fmt: on
    idx = df.x_o.abs() > 0.3
    ax_ratio.plot(df.s[idx], df.x_o[idx] / dfp.x_o[idx], label="$x_o$")
    idx = df.y_o.abs() > 0.3
    ax_ratio.plot(df.s[idx], df.y_o[idx] / dfp.y_o[idx], label="$y_o$")

    for ax in ax_x, ax_y:
        ax.set_ylabel("x,y [mm]")
        ax.legend()

    plt.setp(
        ax_ratio.xaxis.get_ticklabels(),
        "horizontalalignment",
        "right",
        "verticalalignment",
        "top",
        "rotation",
        45,
    )

    # Debug plots
    df = df.sort_index()
    dfp = dfp.sort_index()
    fig, axes = plt.subplots(3, 1, sharex=True)
    ax_x, ax_y, ax_ratio = axes
    # fmt: off
    ax_x.plot(dfp.x_o, marker="x", label="$x_p$")
    ax_y.plot(dfp.y_o, marker="x", label="$y_p$")
    plt.setp(ax_y.get_xticklabels(), "horizontalalignment", "right", "verticalalignment", "top","rotation", 45)
    # plt.setp(ax_y.get_xticklabels(), )
    # plt.setp(ax_y.get_xticklabels(), )

    ax_x.plot(-df.x_o * cooking_factor, marker="+", label="$x_o$")
    ax_y.plot( df.y_o * cooking_factor, marker="+", label="$y_o$")
    # fmt: on
    idx = df.x_o.abs() > 0.3
    ax_ratio.plot(df.x_o[idx] / dfp.x_o[idx], "+", label="$x_o$")
    idx = df.y_o.abs() > 0.3
    ax_ratio.plot(df.y_o[idx] / dfp.y_o[idx], "+", label="$y_o$")

    for ax in ax_x, ax_y:
        ax.set_ylabel("x,y [mm]")
        ax.legend()

    plt.setp(
        ax_ratio.xaxis.get_ticklabels(),
        "horizontalalignment",
        "right",
        "verticalalignment",
        "top",
        "rotation",
        45,
    )


def main(uid):
    # Create a InitializeMongo instance
    mongo_init = InitializeMongo()
    # Get the collection you need
    estimated_angles_collection = mongo_init.get_collection("estimatedangles")
    d = estimated_angles_collection.find_one(dict(uid=uid))
    estimated_angles = EstimatedAngles(
        per_magnet=[jsons.load(m, MagnetEstimatedAngles) for m in d["per_magnet"]],
        md=d["md"],
    )
    return overview_dataframe(estimated_angles)


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    prog_name, uid = sys.argv
    plot_overview_dataframe(main(uid))
    plt.show()
