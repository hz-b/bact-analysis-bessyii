"""Tests to verify that beam based alignment equations work

"""
import copy
import logging
import os

import numpy as np
import pandas as pd
import reportlab.lib.colors
import xarray as xr
import matplotlib.pyplot as plt
import pytest
from pathlib import Path
import thor_scsi.lib as tslib
import gtpsa
from bact_analysis.transverse.process import process_magnet_plane
from bact_analysis.utils.preprocess import rename_doublicates
from numpy import linalg

from thor_scsi.factory import accelerator_from_config
from thor_scsi.utils.accelerator import (
    instrument_with_standard_observers,
    extract_orbit_from_accelerator_with_standard_observers,
)
from thor_scsi.utils.linear_optics import compute_Twiss_along_lattice
from thor_scsi.utils.closed_orbit import compute_closed_orbit
from bact_math_utils import distorted_orbit
from bact_analysis.bba import calc as bba_calc

from bact_math_utils.linear_fit import linear_fit_1d, x_to_cov, cov_to_std

t_file = (
    Path(os.environ["HOME"])
    / "Devel"
    / "gitlab"
    / "dt4acc"
    / "lattices"
    / "b2_stduser_beamports_blm_tracy_corr.lat"
)

logger = logging.getLogger("bact-bba-test")


def create_accelerator():
    """create a new one for each test

    Returns: AcceleratorForTest

    """
    print(f"Reading lattice file {t_file}")
    acc = accelerator_from_config(t_file)
    return acc


def copy_acc(acc):
    """Should be a method, not yet implemented"""
    return tslib.Accelerator([elem for elem in acc])


def copy_quadrupole(quadrupole):
    nmuls = quadrupole.get_multipoles().copy()
    nquad = quadrupole.__class__(quadrupole.config())
    nquad.set_field_interpolator(nmuls)
    nquad.set_dx(quadrupole.get_dx())
    nquad.set_dy(quadrupole.get_dy())
    return nquad


def acc_with_replaced_element(acc, element):
    """But no check if element types match"""
    elements = [elem for elem in acc]
    elements[element.index] = element
    return tslib.Accelerator(elements)


def calculate_required_dipole_compensation(K: complex, dz: complex) -> complex:
    """linear part of Taylor series

    Positive sign as it is intended for compensation
    i.e. a positive shift creates a negative component
    """
    return K * dz


def set_required_dipole_compensation_to_quadrupole(
    quadrupole: tslib.Quadrupole, dz: complex, *, copy: bool = True
):
    if copy:
        quadrupole = quadrupole.copy()

    dx, dy = dz.real, dz.imag
    quadrupole.set_dx(dx)
    quadrupole.set_dy(dy)

    muls = quadrupole.get_multipoles()
    C2 = muls.get_multipole(2)
    C1 = calculate_required_dipole_compensation(C2, dz)
    muls.set_multipole(1, C1)
    quadrupole.set_field_interpolator(muls)
    return quadrupole


def equivalent_angle_for_offset_from_estimate(
    angle: float, offset_for_angle: float, target_offset: float
) -> float:
    return target_offset / offset_for_angle * angle


def angle_for_quadrupole_offset(
    quadrupole: tslib.Quadrupole, target_offset: float, *, angle_for_estimate=1e-3
) -> float:
    quadrupole_info = bba_calc.MagnetInfo(
        length=quadrupole.get_length(),
        tf=abs(quadrupole.get_multipoles().get_multipole(2)),
        polarity=np.sign(quadrupole.get_multipoles().get_multipole(2)),
    )
    assert quadrupole_info.angle_to_offset(0) == pytest.approx(0, abs=1e-12)

    offset_for_angle = quadrupole_info.angle_to_offset(angle_for_estimate)
    equivalent_angle = equivalent_angle_for_offset_from_estimate(
        angle=angle_for_estimate,
        offset_for_angle=offset_for_angle,
        target_offset=target_offset,
    )
    return equivalent_angle


def check_phase_space_zero(ps: gtpsa.ss_vect_double | gtpsa.ss_vect_tpsa):
    """check that constant part of phase space are all zero"""
    assert ps.x == pytest.approx(0, abs=1e-12)
    assert ps.px == pytest.approx(0, abs=1e-12)
    assert ps.y == pytest.approx(0, abs=1e-12)
    assert ps.py == pytest.approx(0, abs=1e-12)
    assert ps.ct == pytest.approx(0, abs=1e-12)
    assert ps.delta == pytest.approx(0, abs=1e-12)


def test_010_create_accelerator():
    acc = create_accelerator()
    assert acc


def test_011_modifiy_copied_muls_without_sideefect_on_original():
    C3 = 355 / 113 * 1j
    C2 = 1.2
    muls = tslib.TwoDimensionalMultipoles(0j)
    muls.set_multipole(3, C3)
    muls_ref = muls.copy()
    muls.set_multipole(2, C2)

    # Check that the multipole was set
    assert muls.get_multipole(2).real == pytest.approx(C2.real, abs=1e-12)
    assert muls.get_multipole(2).imag == pytest.approx(C2.imag, abs=1e-12)
    assert muls.get_multipole(3).real == pytest.approx(C3.real, abs=1e-12)
    assert muls.get_multipole(3).real == pytest.approx(C3.real, abs=1e-12)

    # Check that the original was not modified
    assert muls_ref.get_multipole(2).real == pytest.approx(0, abs=1e-12)
    assert muls_ref.get_multipole(2).imag == pytest.approx(0, abs=1e-12)
    assert muls_ref.get_multipole(3).real == pytest.approx(C3.real, abs=1e-12)
    assert muls_ref.get_multipole(3).imag == pytest.approx(C3.imag, abs=1e-12)


def test_012_accelerator_modifiy_quad_without_sideeffect_on_orginal():
    acc_ref = create_accelerator()
    quadrupole_name = "q3m2d4r"
    quadrupole = acc_ref.find(quadrupole_name, 0)

    nquad = copy_quadrupole(quadrupole)
    K = nquad.get_main_multipole_strength()
    modified_K = K * (1 + 2.3e-2)
    nquad.get_multipoles().set_multipole(2, modified_K)
    acc = acc_with_replaced_element(acc_ref, nquad)

    # check that original quadrupole has not been modified
    assert acc_ref.find(
        quadrupole_name, 0
    ).get_main_multipole_strength() == pytest.approx(K, abs=1e-12)

    # check that quadrupole was modified for the second accelerator
    assert acc.find(quadrupole_name, 0).get_main_multipole_strength() == pytest.approx(
        modified_K, abs=1e-12
    )


def test_020_fixed_point_default():
    """Default fixed point at 0..0 without distortions"""
    acc = create_accelerator()
    calc_info = tslib.ConfigType()

    ps = gtpsa.ss_vect_double(0e0)
    acc.propagate(calc_info, ps)
    check_phase_space_zero(ps)


def test_030_dipole_compensation_of_moved_quadrupole():
    """Test that dipole component is compensated for moved quad"""
    acc = create_accelerator()
    quadrupole_name = "q4m1t2r"
    quadrupole = acc.find(quadrupole_name, 0)

    dy = 3e-4
    quadrupole = set_required_dipole_compensation_to_quadrupole(
        quadrupole, dy * 1j, copy=False
    )
    # Check field interpolation

    # Check that quadrupole setup worked
    # offset
    assert quadrupole.get_dy() == pytest.approx(dy, abs=1e-12)
    # dipole component
    C1 = calculate_required_dipole_compensation(
        quadrupole.get_multipoles().get_multipole(2), dy * 1j
    )

    assert quadrupole.get_multipoles().get_multipole(1).real == pytest.approx(
        C1.real, abs=1e-12
    )
    assert quadrupole.get_multipoles().get_multipole(1).imag == pytest.approx(
        C1.imag, abs=1e-12
    )

    # Check that field interpolation is as expected
    muls = quadrupole.get_multipoles()
    field = muls.field(0)
    assert field.real == pytest.approx(C1.real, abs=1e-12)
    assert field.imag == pytest.approx(C1.imag, abs=1e-12)

    calc_info = tslib.ConfigType()
    ps = gtpsa.ss_vect_double(0e0)
    quadrupole.propagate(calc_info, ps)
    check_phase_space_zero(ps)


def test_031_fixed_point_default_with_moved_compensated_quadrupole():
    """Move a quadrupole and compensate with its dipole component"""
    acc = create_accelerator()
    quadrupole_name = "q4m1t2r"
    quadrupole = acc.find(quadrupole_name, 0)
    dy = 1e-4  # m

    set_required_dipole_compensation_to_quadrupole(quadrupole, dy * 1j, copy=False)

    calc_info = tslib.ConfigType()

    ps = gtpsa.ss_vect_double(0e0)
    acc.propagate(calc_info, ps)
    check_phase_space_zero(ps)


def test_040_test_angle_for_offset():
    dy = 355 / 113 * 1e-4
    acc = create_accelerator()
    quadrupole_name = "q4m1t2r"
    quadrupole = acc.find(quadrupole_name, 0)
    equivalent_angle = angle_for_quadrupole_offset(quadrupole, dy)
    txt = f"dy {dy * 1000:.3f} mm equivalent angle {equivalent_angle*1e6:.3f} urad"
    print(txt)

    info = bba_calc.MagnetInfo(
        length=quadrupole.get_length(),
        tf=abs(quadrupole.get_main_multipole_strength()),
        polarity=np.sign(quadrupole.get_main_multipole_strength()),
    )
    # Estimate the angle for a given offset ... linear problem
    assert info.angle_to_offset(equivalent_angle) == pytest.approx(dy, abs=1e-12)


def closed_orbit_distortion_from_twiss(
    *, twiss_db: xr.Dataset, quadrupole_index: int, plane: str, angle: float
) -> np.ndarray:
    """compute closed orbit using twiss data as provided by thor scsi"""
    nu = xr.apply_ufunc(np.add.accumulate, twiss_db.twiss.sel(par="dnu"))
    logger.info("Working point: ", nu.isel(index=-1).values)

    orbit = distorted_orbit.closed_orbit_distortion(
        beta=twiss_db.twiss.sel(plane=plane, par="beta").values,
        mu=nu.sel(plane=plane).values * 2 * np.pi,
        # Todo Do I need to divide with 2 * pi ?
        tune=nu.sel(plane=plane).isel(index=-1).values,
        beta_i=twiss_db.twiss.sel(
            plane=plane, par="beta", index=quadrupole_index
        ).values,
        mu_i=nu.sel(plane="y", index=quadrupole_index).values * 2 * np.pi,
        theta_i=angle,
    )
    return orbit


def closed_orbit_distortion_from_model(
    acc,
    quadrupole_name: str,
    relative_gradient_change: float | None = None,
    gradient: float | None = None,
    copy: bool = True,
) -> [float, xr.Dataset]:
    """Compute closed orbit from model.

    Warning:
        Copy not yet working

    Todo:
        Current checks assume that change happens in y plane
        Make it side effect free: i.e. changes to the quadrupole in
        this lattice should not effect any other lattice
    """
    if copy:
        acc = copy_acc(acc)

    quadrupole = acc.find(quadrupole_name, 0)
    if copy:
        quadrupole = copy_quadrupole(quadrupole)

    # change quadrupole ... after reference orbit has been computed
    # as for analysis: reference orbits computed for ideal machine
    K2 = quadrupole.get_main_multipole_strength()
    if gradient:
        modified_K2 = gradient
        dG = modified_K2 - K2
    else:
        assert relative_gradient_change
        modified_K2 = K2 * (1 + relative_gradient_change)
        dG = K2 * relative_gradient_change
    muls = quadrupole.get_multipoles()
    if copy:
        muls = muls.copy()
    muls.set_multipole(2, modified_K2)
    quadrupole.set_field_interpolator(muls)

    # todo: only if copy requested =
    acc = acc_with_replaced_element(acc, quadrupole)

    assert quadrupole.get_main_multipole_strength().real == pytest.approx(
        modified_K2, rel=1e-12
    )
    # and if retrieved from the machine
    assert acc.find(
        quadrupole_name, 0
    ).get_main_multipole_strength().real == pytest.approx(modified_K2, rel=1e-12)
    # compute the closed orbit
    ob = instrument_with_standard_observers(acc, mapping=gtpsa.default_mapping())
    calc_config = tslib.ConfigType()

    r = compute_closed_orbit(acc, calc_config, delta=0, max_iter=100, eps=1e-15)
    assert r.found_closed_orbit
    # no effect in x plane expected ... to be small
    print("Quadrupole modified: closed orbit fixed point", r.x0)
    assert r.x0.x == pytest.approx(0e0, abs=2e-5)
    assert r.x0.px == pytest.approx(0e0, abs=1e-6)

    ds = extract_orbit_from_accelerator_with_standard_observers(acc)
    return modified_K2, ds


# @pytest.mark.skip
@pytest.mark.parametrize(
    ["quadrupole_name", "bba_gradient_change"],
    (
        # ["q4m2t2r", -1e-2],
        # ["q4m2t2r", 1e-2],
        # q2 strong in y direction .. large beta_y
        # ["q2m1t8r", -1e-2],
        # ["q2m2t8r", 1e-2],
        # q3 strong in y direction .. large beta_y
        ["q3m1t8r", -1e-2],
        ["q3m2t8r", 1e-2],
    ),
)
def test_050_predicted_deviation_to_closed_orbit(
    quadrupole_name, bba_gradient_change, do_plots: bool = True
):
    """A nearly complete bba test

    Test for a moved quadrupole that the beam deviation is as expected

    1. move quadrupole
    2. change K value by known fraction
    3. calculate predicted orbit change
    4. compare that deviation is as predicted

    Todo:
        make comparisons how well the model fits if only the Bpm's are used
    """
    # parameters --------------------
    # parameters
    quad_dy = 1e-4  # m
    # change gradient by fraction
    # bba_gradient_change = -7e-2
    # quadrupole_name = "q4m2t2r"
    # quadrupole_name = "q1m1t1r"
    # ----------------------------------

    # use accelerator to
    # 1. calculate reference distorted orbit
    # 2. compare computed closed orbit to distorted orbit
    acc = create_accelerator()
    # check if a better prediction is reached with more integration steps
    # for elem in acc:
    #    if isinstance(elem, tslib.Bending):
    #        elem.set_number_of_integration_steps(10)
    #    if isinstance(elem, tslib.Quadrupole):
    #        # 2 step per 0.1 m
    #        elem.set_number_of_integration_steps(2 * int(np.ceil(elem.get_length() / 0.1)))
    quadrupole = acc.find(quadrupole_name, 0)

    # shift quadrupole and compensate its offset with a steerer
    set_required_dipole_compensation_to_quadrupole(quadrupole, quad_dy * 1j, copy=False)
    desc = gtpsa.desc(6, 2)
    db = compute_Twiss_along_lattice(2, acc, desc=desc, mapping=gtpsa.default_mapping())

    equivalent_angle = angle_for_quadrupole_offset(quadrupole, quad_dy * 1j)
    orbit = closed_orbit_distortion_from_twiss(
        twiss_db=db,
        quadrupole_index=quadrupole.index,
        plane="y",
        angle=equivalent_angle.imag,
    )

    # change quadrupole ... after reference orbit has been computed
    # as for analysis: reference orbits computed for ideal machine
    _, ds = closed_orbit_distortion_from_model(
        acc,
        quadrupole_name,
        relative_gradient_change=bba_gradient_change,
        copy=False,
    )
    orbit_y = ds.ps.sel(phase_coordinate="y")

    # effect in y plane expected: lattice dependent
    # assert abs(r.x0.y) > .5e-4
    # assert abs(r.x0.py) > 1e-5

    #: todo .. why the minus here
    X = (-orbit * bba_gradient_change)[:, np.newaxis]
    cooking_factor, residues, rank, s = linalg.lstsq(X, orbit_y.values, rcond=None)
    cooking_factor = float(cooking_factor)
    assert rank == 1
    cooking_factor_error = float(cov_to_std(x_to_cov(X, residues, len(orbit_y), 1)))
    del X, rank, s

    # todo: where does the cooking factor come from!
    # todo: assumption of the approximation only small quads?
    #    assert cooking_factor == pytest.approx(1, rel=0.2)
    #    assert cooking_factor_error == pytest.approx(0, abs=1e-3)

    p, dp = cooking_factor - 1, cooking_factor_error
    print(dp)
    print(
        f"for {quadrupole_name} bba changed gradient by  {bba_gradient_change}"
        f"\nneed to adjust forecast by {p*100:.4f} +/- {dp*100:.4f} %"
    )

    # Make some statistical computations to get an estimate how much the orbit
    # is off
    # done twice: once when the cooking factor was used
    # and once without it
    orbit_y_ref = -orbit * bba_gradient_change * cooking_factor
    dy = orbit_y - orbit_y_ref
    mean_square_error = np.sum(dy ** 2)
    mean_absolute_error = np.sum(np.absolute(dy))

    # in average the prediction is allowed to be wrong by 10 microns for
    # each element position .. divide by element to get an average feeling
    n_elements = len(acc)
    assert mean_square_error / n_elements < 10e-6
    assert mean_absolute_error / n_elements < 10e-6
    del orbit_y_ref, dy, mean_square_error, mean_absolute_error

    # Compare offset
    orbit_y_expected = -orbit * bba_gradient_change
    orbit_y_fit = -orbit * bba_gradient_change * cooking_factor
    dy_expected = orbit_y - orbit_y_expected
    dy_fit = orbit_y - orbit_y_fit

    # For offset from model
    mean_square_error = np.mean(dy_expected ** 2)
    mean_absolute_error = np.mean(np.absolute(dy_expected))
    # in average the prediction is allowed to be wrong by 10 microns for
    # each element position .. divide by element to get an average feeling
    print(f"Difference to expectation MSE{mean_square_error} MAE{mean_absolute_error}")
    assert mean_square_error < 10e-6
    # needs roughly 10 times more
    assert mean_absolute_error < 10 * 10e-6

    # For offset from adjusted model
    mean_square_error = np.mean(dy_fit ** 2)
    mean_absolute_error = np.mean(np.absolute(dy_fit))
    # in average the prediction is allowed to be wrong by 10 microns for
    # each element position .. divide by element to get an average feeling
    print(f"Difference to adjusted MSE{mean_square_error} MAE{mean_absolute_error}")

    assert mean_square_error / n_elements < 10e-6
    # needs roughly 10 times more
    assert mean_absolute_error / n_elements < 10 * 10e-6

    if not do_plots:
        # No plots requested
        return

    # scaling plot
    # todo: negative sign for vertical plane ... where to handle it properly
    orbit_y_expected = -orbit * bba_gradient_change

    pscale = 1e6  # m to um
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax, ax_diff = axes
    ax.plot(db.s, orbit_y * pscale, "k-", label="closed orbit")
    (line_twiss,) = ax.plot(
        db.s, orbit_y_expected * pscale, "-", label="Forcast from twiss"
    )
    (line_twiss_scaled,) = ax.plot(
        db.s,
        orbit_y_fit * pscale,
        "-",
        label=f"Forcast from twiss scaled by {p * 100:.3f} +/- {dp * 100:.3f} %",
    )
    ax.legend()
    ax.set_ylabel("y [um]")
    ax.set_title(f"Quadrupole {quadrupole.name} offset dy {quad_dy*1000:.3f} mm")

    ax_diff.plot(
        db.s,
        dy_expected * pscale,
        color=line_twiss.get_color(),
        label="difference to twiss estimate",
    )
    ax_diff.plot(
        db.s,
        dy_fit * pscale,
        color=line_twiss_scaled.get_color(),
        label=f"Forcast from twiss",
    )
    ax_diff.set_xlabel("s [m]")
    ax_diff.set_ylabel("y [um]")
    ax_diff.legend()
    #
    plt.show()


# @pytest.mark.skip
@pytest.mark.parametrize(
    ["quadrupole_name", "bba_gradient_change", "test_angle_fraction"],
    (
        # q2 strong in y direction .. large beta_y
        ["q3m2t8r", 0.23e-2, 0.223],
        ["q4m1t8r", 0.105e-2, 2 * 355/113.0],
    ),
)
def test_060_predicted_deviation_to_closed_orbit_similar_to_measurement(
    quadrupole_name, bba_gradient_change, test_angle_fraction, do_plots: bool = True
):
    """Test if fit to measurement yields expected data

    Todo:
        check forecast if estimated angle is not close to
        the expected one
    """

    acc = create_accelerator()
    desc = gtpsa.desc(6, 2)
    quad_dy = 1e-4

    quadrupole = acc.find(quadrupole_name, 0)
    # keep the reference ... check that changing the gradient had the
    # desired effect
    k_ref = quadrupole.get_main_multipole_strength()
    # shift quadrupole and compensate its offset with a steerer
    set_required_dipole_compensation_to_quadrupole(quadrupole, quad_dy * 1j, copy=False)

    acc_ref = copy_acc(acc)

    twiss_db = compute_Twiss_along_lattice(
        2, acc_ref, desc=desc, mapping=gtpsa.default_mapping()
    )
    equivalent_angle = angle_for_quadrupole_offset(quadrupole, quad_dy * 1j)
    # need to scale the angle: from full dipole to the part used by
    # gradient change during bba
    equivalent_angle = equivalent_angle * bba_gradient_change
    # calculate orbit from twiss ... check what internal routines are doing
    orbit_twiss = closed_orbit_distortion_from_twiss(
        twiss_db=twiss_db,
        quadrupole_index=quadrupole.index,
        plane="y",
        # scale the angle by the equivalent angle
        # equivalent angle will give a fit result close to 1
        # here it is deliveratly forced to be an other value
        angle=equivalent_angle.imag,
    )

    # prepare data as processing expects it
    tmp = twiss_db.twiss.sel(plane="y")
    beta = tmp.sel(par="beta")
    beta.name = "beta"
    mu = xr.apply_ufunc(np.add.accumulate, tmp.sel(par="dnu"))
    mu.name = "mu"
    selected_model = xr.merge([beta, mu], compat="override")
    del tmp, beta, mu

    # prepare the dimension index to pos_name using unique names: lookup of element by name
    _, pos_names = rename_doublicates([elem.name for elem in acc])
    selected_model = selected_model.assign_coords(index=pos_names).rename(index="pos")
    selected_model_for_magnet = selected_model.sel(pos=quadrupole.name)

    # todo: check if process should not ensure that arguments are arrays
    excitation = np.array([-bba_gradient_change, bba_gradient_change])
    excitation_da = xr.DataArray(
        data=excitation, dims=["dG"], coords=[excitation], name="exitation"
    )
    # precompute the gradients: changing the model
    # not yet betting that copies of accelerator lattice are properly made
    gradients = k_ref * (1 + excitation)

    # copy of lattice and lattice elements: is it already working ?
    def calculate_closed_orbit_for_gradient(gradient):
        acc = copy_acc(acc_ref)
        modified_K, ds = closed_orbit_distortion_from_model(
            acc, quadrupole_name, gradient=gradient, copy=False
        )
        return modified_K, ds.ps.sel(phase_coordinate="y")

    bpm_names = [name for name in pos_names if name[:3] == "bpm"]
    print(bpm_names)
    tmp = [calculate_closed_orbit_for_gradient(g) for g in gradients]
    used_gradients = [t[0] for t in tmp]
    dG = excitation_da * k_ref.real
    # check that the gradients are as expected
    for used_gradient, ex in zip(used_gradients, excitation_da.values):
        expected_gradient = k_ref * (1 + ex)
        print(
            f"original gradient {k_ref} -> expected {expected_gradient} found  {used_gradient}"
        )
        assert used_gradient == pytest.approx(expected_gradient, rel=1e-12)
    # testing the effect of excitation
    # dG = np.array([-1, 1])
    dG_da = xr.DataArray(data=dG, dims=["dG"], coords=[dG], name="exitation")

    offset = xr.DataArray(
        data=[t[1] for t in tmp],
        name="measured_offset",
        dims=["dG", "pos"],
        coords=[dG.values, pos_names],
    )
    del tmp
    # in real world the offsets are only available at the beam position monitors
    offset = offset.sel(pos=bpm_names)
    r = process_magnet_plane(
        selected_model=selected_model,
        selected_model_for_magnet=selected_model_for_magnet,
        excitation=dG_da,
        offset=offset,
        bpm_names=bpm_names,
        # scale the angle by the equivalent angle
        # equivalent angle will give a fit result close to 1
        # here it is deliveratly forced to be an other value
        theta=equivalent_angle.imag * test_angle_fraction,
        scale_phase_advance=2 * np.pi,
    )

    # Check that the correct angle is stored
    r.orbit.attrs["theta"] == pytest.approx(
        equivalent_angle.imag * test_angle_fraction, 1e-12
    )
    r.orbit_at_bpm.attrs["theta"] == pytest.approx(
        equivalent_angle.imag * test_angle_fraction, 1e-12
    )

    scaled_angle = r.result.sel(parameter="scaled_angle")

    angle_slope = scaled_angle * r.orbit.attrs["theta"]
    # The slope of the angle is equivalent to the kick that
    # is produced by the dipole created by feed down
    # but need to divide by the magnet length ...
    # the integral effect is measued from which one has to conclude
    # to the equivalent dipole strength
    dy_from_fit = -angle_slope / quadrupole.get_length()
    print(dy_from_fit / quad_dy)
    # 5 percent would be enought for the average BESSY II quad
    # but not for the long Q4 ones
    assert dy_from_fit.sel(result="value") == pytest.approx(quad_dy, rel=7e-2)

    # Check that the estimated values are similar to the expected ones
    fit_equivalent_angle = angle_slope * dG
    assert (
        np.absolute(fit_equivalent_angle.sel(result="value"))
        == pytest.approx(np.absolute(equivalent_angle), rel=7e-2)
    ).all()

    # bpm_offsets = r.result.sel(parameter=bpm_names)
    # # ALl bpm offsets should be now close to zero
    # for name in bpm_names:
    #     chk = bpm_offsets.sel(parameter=name)
    #     assert chk.sel(result="value").values == pytest.approx(
    #         0, abs=chk.sel(result="error").values
    #     )

    do_plots = True
    if not do_plots:
        return

    print("scaled angle\n", scaled_angle)
    print("scaled angle * dG\n", scaled_angle * dG)
    print("scaled angle / expected_angle\n", scaled_angle / equivalent_angle.imag)

    fig, ax = plt.subplots(1, 1)
    pscale = 1e6

    # if names are not unique ... work around
    s_bpm = np.array(
        [twiss_db.s.isel(index=twiss_db.names == name) for name in bpm_names]
    )

    # plot first the orbit as used for the fit

    polarity = np.sign(k_ref)
    for dg in dG.values:
        if (dg) < 0:
            s = -1 * polarity
        else:
            s = 1 * polarity

        # fmt: off
        # as estimated by twiss calculation
        line, = ax.plot(twiss_db.s, - orbit_twiss * pscale, "-",
            label="orbit deviation estimated from twiss")
        # as found by closed orbit finder
        ax.plot(s_bpm, offset.sel(dG=dg) * pscale * s, ".", label=f"dG ={dg:.3f}",
                color=line.get_color(), linewidth=0.5,
        )
        # scaled orbit matching data
        scale_fit = (
            # the scaled angle is retrieved from a linear fit
            # thus the scaled angle is for the equivalent angle
            # when the excitation is at 1
            scaled_angle.sel(result="value")
            # here the excitation was dG .. so it needs to be
            # multiplied with this value
            * dg
        )
        scaled_fit = abs(scale_fit)
        print(f"scale used orbit data with {scaled_fit}")
        ax.plot(s_bpm, - r.orbit_at_bpm *  pscale, "+",
                label=f"orbit bpm data used to scaled to closed orbit data ={dg:.3f}",
                color=line.get_color(), linewidth=0.5,
                )
        ax.plot(twiss_db.s, - r.orbit * pscale * scaled_fit, "--",
                label="orbit used for estimating angle scale",
                color=line.get_color(), linewidth=0.5,
                )
        # ax.plot(s_bpm, - r.orbit_at_bpm * pscale, "k.", label="for bpm")

        # fmt: on

    ax.legend()
    ax.set_ylabel("dy [um]")
    ax.set_xlabel("s [m]")
    plt.show()
