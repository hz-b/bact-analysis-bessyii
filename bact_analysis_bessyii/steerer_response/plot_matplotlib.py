import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from bact_math_utils.misc import EnumerateUniqueEverSeen
from dt4acc.model.twiss import Twiss

from .model import OrbitPredictionForKicks, OrbitPredictionCollection, AcceleratorDescription
from ..model.analysis_model import EstimatedAngles, FitReadyDataPerMagnet, FitReadyData, FitResult, \
    MagnetEstimatedAngles, MeasuredValues


def plot_twiss_functions(twiss: Twiss, s: Sequence[float]):
    fig, axes = plt.subplots(2, 1)
    ax_x, ax_y = axes

    q_scale =  1 / (2 * np.pi)
    line, = ax_x.plot(twiss.x.nu * q_scale, twiss.x.beta, '-')
    ax_y.plot(twiss.y.nu * q_scale, twiss.y.beta, '-', color=line.get_color())
    ax_x.set_ylabel(r"$\beta_x$ [m]")
    ax_y.set_ylabel(r"$\beta_y$ [m]")
    ax_y.set_xlabel("s [m}")

    fig, axes = plt.subplots(2, 1, sharex=True)
    ax_x, ax_y = axes
    ax_x.plot(s, twiss.x.beta, '-', color=line.get_color())
    ax_y.plot(s, twiss.y.beta, '-', color=line.get_color())
    ax_x.set_ylabel(r"$\beta_x$ [m]")
    ax_y.set_ylabel(r"$\beta_y$ [m]")
    ax_y.set_xlabel("s [m}")

    fig, axes = plt.subplots(2, 1, sharex=True)
    ax_x, ax_y = axes
    q_scale =  1 / (2 * np.pi)
    ax_x.plot(s, twiss.x.nu * q_scale, '-', color=line.get_color())
    ax_y.plot(s, twiss.y.nu * q_scale, '-', color=line.get_color())
    ax_x.set_ylabel("$Q_x$")
    ax_y.set_ylabel("$Q_y$")
    ax_y.set_xlabel("s [m}")


def plot_forecast_difference(
    measured_data: FitReadyData,
    orbit_prediction: OrbitPredictionCollection,
    fits : EstimatedAngles,
    acc_desc : AcceleratorDescription,
):
    fig, axes_basis = plt.subplots(2, 1, sharex=True)
    fig, axes_relative = plt.subplots(2, 1, sharex=True)
    fig, axes_comparison = plt.subplots(2, 1, sharex=True)


    # ["HS1PD6R", 'VS2P2D4R', 'VS2P2D7R', 'VS2P2D1R']
    sys.stdout.write("\n")
    for measurement_datum in measured_data.per_magnet[::-1]:
        magnet_name = measurement_datum.name

        t_fit = fits.get(magnet_name)

        def mse_mae(error_estimates):
            p_scale = 1e6
            return (
                np.sqrt(np.sum(error_estimates.mean_square_error)) * p_scale,
                np.mean(np.absolute(error_estimates.mean_absolute_error)) * p_scale
            )
        mse_x, mae_x = mse_mae(t_fit.x.error_estimates)
        mse_y, mae_y = mse_mae(t_fit.x.error_estimates)
        sys.stdout.write(
            f"creating plots for {magnet_name:10s}"
            f" x mse {mse_x:4.0f} mae {mae_x:4.0f}"
            f" y mse {mse_y:4.0f} mae {mae_y:4.0f}"
            " ... "
        )
        for axes in axes_basis, axes_relative, axes_comparison:
            for ax in axes:
                ax.clear()
        sys.stdout.flush()
        plot_forecast_difference_for_magnet(
            measured_data.get(magnet_name),
            orbit_prediction.get(magnet_name),
            fits.get(magnet_name),
            acc_desc,
            axes=[axes_basis, axes_relative, axes_comparison]
        )
        input("done")



def plot_forecast_difference_for_magnet(
        measured_data: FitReadyDataPerMagnet,
        orbit_prediction: OrbitPredictionForKicks,
        fit_result: MagnetEstimatedAngles,
        acc_desc: AcceleratorDescription,
        axes: Sequence[plt.Axes]
):
    assert measured_data.name == orbit_prediction.name == fit_result.name

    bpm_names = [datum.name for datum in measured_data.x[0].data]
    for n1, n2, n3 in zip(
        bpm_names,
        [datum.name for datum in fit_result.x.bpm_offsets],
        [datum.name for datum in orbit_prediction.x[0].orbit]
    ):
        if n1 != n2 or n1 != n3:
            print(f"BPM pos names not matching for {n1}, {n2}, {n3}")

    s = [acc_desc.survey.get(name).value for name in bpm_names]

    axes_basis, axes_relative, axes_comparison = axes
    ax_x, ax_y, = axes_basis
    ax_diff, ay_diff = axes_relative
    ax_test, ay_test = axes_comparison

    for measurements, orb4p, fr4p, axes in zip(
        [measured_data.x, measured_data.y],
        [orbit_prediction.x, orbit_prediction.y],
        [fit_result.x, fit_result.y],
        [(ax_x, ax_diff, ax_test), (ax_y, ay_diff, ay_test)]
    ):

        ax, a_diff, a_test = axes
        bpm_offsets = np.array([datum.value for datum in fr4p.bpm_offsets])

        # todo: need to handle fit of 1D and 2D data
        # for excitation, color, m, orb_exc  , A_exc, b_exc in zip(
        for excitation, color, m, orb_exc in zip(
            measured_data.excitations,
            color_for_excitation(measured_data.excitations),
            measurements,
            orb4p,
            # fr4p.equivalent_angle.input.A[:, :, -1],
            # fr4p.equivalent_angle.input.b,
        ):

            orb = np.array([datum.value for datum in orb_exc.orbit])
            mea = np.array([datum.value for datum in m.data])

            # plot scale
            p_scale = 1e3

            # Data points as measured and used in fit
            ax.plot(s, mea * p_scale, '.', color=color, label="measurement")
            ax.plot(s, (bpm_offsets) * p_scale, '--', color=color, label="bpm_offsets")
            ax.plot(s, (orb + bpm_offsets) * p_scale, '-', color=color, label="predicted orbit")
            # ax.plot(s, b_exc * p_scale, 'x', color=color, label="measured data used for fit")

            # Data as above but with bpm offsets subtracted
            a_diff.plot(s, (mea - bpm_offsets) * p_scale, color=color, label="measurement", linestyle='-', linewidth=.2, marker='.')
            a_diff.plot(s, orb * p_scale, '.-', color=color, label="orbit inferred from fit")
            # a_diff.plot(s, (b_exc - bpm_offsets) * p_scale, 'x', color=color, label="measured data used for fit - bpm offset")

            # Orbit used at fit
            # a_test.plot(s, A_exc * p_scale, '-', color=color, label="scaled orbit used for fit")
            # a_test.plot(s, A_exc * p_scale * fr4p.equivalent_angle.value , '--', color=color, label="orbit as expected to match fit")

    ax_x.set_title(f"Magnet {measured_data.name}")
    ax_x.set_ylabel("x [mm]")
    ax_y.set_ylabel("y [mm]")

    ax_diff.set_title(f"Magnet {measured_data.name} offset substracted")
    ax_diff.set_ylabel(r"$\Delta$ x [mm]")
    ay_diff.set_ylabel(r"$\Delta$ y [mm]")

    ax_test.set_title(f"Magnet {measured_data.name}: orbit distortion used for fit")
    ax_test.set_ylabel("x [mm]")
    ay_test.set_ylabel("y [mm]")

    for ax in ax_y, ay_diff, ay_test:
        ax.set_xticks(s)
        ax.set_xticklabels(bpm_names)
        plt.setp(ax.get_xticklabels(), "verticalalignment", "top")
        plt.setp(ax.get_xticklabels(), "horizontalalignment", "right")
        plt.setp(ax.get_xticklabels(), 'fontsize', 'small')
        plt.setp(ax.get_xticklabels(), 'rotation', 45)


def color_for_excitation(excitations):
    colors = ['r', 'b', 'g', 'k', 'c', 'm']
    for num, excitation in EnumerateUniqueEverSeen()(excitations):
        yield colors[num]


def measurements_sorted_by_position(measurements: Sequence[MeasuredValues]):
    def extract(measurement: MeasuredValues, pos_name: str):
        item  = measurement.get(pos_name)
        return item.value, item.rms
    
    pos_names = [datum.name for datum in measurements[0].data]
    return np.array(
        [
            [extract(measurement, name) for measurement in measurements]
             for name in pos_names
        ]
    )


def plot_bpm_offsets(measured_data: FitReadyData, data: EstimatedAngles, acc_desc: AcceleratorDescription
):

    fig,  axes = plt.subplots(2, 1)
    ax_x, ax_y = axes

    fig,  axes_test = plt.subplots(2, 1)
    ax_test, ay_test = axes_test

    p_scale = 1000
    for meas_for_magnet, data_for_magnet in zip(measured_data.per_magnet, data.per_magnet):
        assert meas_for_magnet.name == data_for_magnet.name

        bpm_names = [datum.name for datum in meas_for_magnet.x[0].data]
        s = [acc_desc.survey.get(name).value for name in bpm_names]

        meas_x_vals = measurements_sorted_by_position(meas_for_magnet.x)
        meas_y_vals = measurements_sorted_by_position(meas_for_magnet.y)
        
        meas_bpm_offsets_x = np.sum(meas_x_vals[...,0], axis=1)
        meas_bpm_offsets_y = np.sum(meas_y_vals[...,0], axis=1)

        # Todo: use accelerator info
        if meas_for_magnet.name[0] == "H":
            meas_for_plane = meas_bpm_offsets_x
            data_for_plane = data_for_magnet.x
            ax = ax_x
            axt = ax_test
        elif meas_for_magnet.name[0] == "V":
            meas_for_plane = meas_bpm_offsets_y
            data_for_plane = data_for_magnet.y
            ax = ax_y
            axt = ay_test
        else:
            raise AssertionError(f"Can'T deduce steerer plane from {meas_for_magnet.name}")

        # line, = ax.plot(s, meas_for_plane * p_scale, '--')
        line, *bars = ax.errorbar(
            s,
            np.array([r.value for r in data_for_plane.bpm_offsets]) * p_scale,
            yerr=np.array([r.std for r in data_for_plane.bpm_offsets]) * p_scale,
            #color=line.get_color()
        )
        axt.plot(s, meas_for_plane * p_scale, '--', color=line.get_color())

    ax_y.set_ylabel("y [mm]")
    ax_x.set_xlabel("x [mm]")
    ay_test.set_ylabel("y [mm]")
    ax_test.set_xlabel("x [mm]")

    ax_x.set_title("bpm offsets from fit")
    ax_test.set_title("bpm offset estimated from average")
