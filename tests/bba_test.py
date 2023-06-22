"""Tests to verify that beam based alignment equations work

"""
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest
from pathlib import Path
import thor_scsi.lib as tslib
import gtpsa
from numpy.linalg import lstsq

# from scipy.linalg import lstsq
from thor_scsi.factory import accelerator_from_config
from thor_scsi.utils.accelerator import (
    instrument_with_standard_observers,
    extract_orbit_from_accelerator_with_standard_observers,
)
from thor_scsi.utils.linear_optics import compute_Twiss_along_lattice
from thor_scsi.utils.closed_orbit import compute_closed_orbit
from bact_math_utils import distorted_orbit
from bact_analysis.bba import calc as bba_calc

t_file = (
    Path(os.environ["HOME"])
    / "Devel"
    / "gitlab"
    / "dt4acc"
    / "lattices"
    / "b2_stduser_beamports_blm_tracy_corr.lat"
)


def create_accelerator():
    """create a new one for each test

    Returns: AcceleratorForTest

    """
    print(f"Reading lattice file {t_file}")
    acc = accelerator_from_config(t_file)
    return acc


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


# @pytest.mark.skip
@pytest.mark.parametrize(
    ["quadrupole_name", "bba_gradient_change"],
    (
            ["q4m2t2r", -1e-2],
            ["q4m2t2r", 1e-2],
            ["q1m1t8r", -1e-2],
            ["q1m2t8r", 1e-2],
    ),
)
def test_050_predicted_deviation_to_closed_orbit(quadrupole_name, bba_gradient_change):
    """A nearly complete bba test

    Test for a moved quadrupole that the beam deviation is as expected

    1. move quadrupole
    2. change K value by known fraction
    3. calculate predicted orbit change
    4. compare that deviation is as predicted
    """
    # parameters --------------------
    # parameters
    dy = 2.3e-4  # m
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
    for elem in acc:
        if isinstance(elem, tslib.Bending):
            elem.set_number_of_integration_steps(10)
            pass
    quadrupole = acc.find(quadrupole_name, 0)

    # shift quadrupole and compensate its offset with a steerer
    set_required_dipole_compensation_to_quadrupole(quadrupole, dy * 1j, copy=False)
    desc = gtpsa.desc(6, 2)
    db = compute_Twiss_along_lattice(2, acc, desc=desc, mapping=gtpsa.default_mapping())

    nu = xr.apply_ufunc(np.add.accumulate, db.twiss.sel(par="dnu"))
    print("Working point: ", nu.isel(index=-1))

    equivalent_angle = angle_for_quadrupole_offset(quadrupole, dy * 1j)

    orbit = distorted_orbit.closed_orbit_distortion(
        beta=db.twiss.sel(plane="y", par="beta").values,
        mu=nu.sel(plane="y").values * 2 * np.pi,
        # Todo Do I need to divide with 2 * pi ?
        tune=nu.sel(plane="y").isel(index=-1).values,
        beta_i=db.twiss.sel(plane="y", par="beta", index=quadrupole.index).values,
        mu_i=nu.sel(plane="y", index=quadrupole.index).values * 2 * np.pi,
        theta_i=equivalent_angle.imag,
    )

    # change quadrupole ... after reference orbit has been computed
    # as for analysis: reference orbits computed for ideal machine
    K2 = quadrupole.get_main_multipole_strength()
    modified_K2 = K2 * (1 + bba_gradient_change)
    quadrupole.get_multipoles().set_multipole(2, modified_K2)
    assert quadrupole.get_main_multipole_strength().real == pytest.approx(
        modified_K2, rel=1e-12
    )
    # and if retrieved from the machine
    assert acc.find(quadrupole_name, 0).get_main_multipole_strength().real == pytest.approx(
        modified_K2, rel=1e-12
    )
    # compute the closed orbit
    ob = instrument_with_standard_observers(acc, mapping=gtpsa.default_mapping())
    calc_config = tslib.ConfigType()

    r = compute_closed_orbit(acc, calc_config, delta=0, max_iter=100, eps=1e-15)
    assert r.found_closed_orbit
    ds = extract_orbit_from_accelerator_with_standard_observers(acc)
    orbit_y = ds.ps.sel(phase_coordinate="y")

    # no effect in x plane expected ... to be small
    print("Quadrupole modified: closed orbit fixed point", r.x0)
    assert r.x0.x == pytest.approx(0e0, abs=2e-5)
    assert r.x0.px == pytest.approx(0e0, abs=1e-6)
    # effect in y plane expected: lattice dependent
    # assert abs(r.x0.y) > .5e-4
    # assert abs(r.x0.py) > 1e-5

    X = np.zeros([orbit_y.shape[0], 1], float)
    X[:, 0] = -orbit * bba_gradient_change
    cooking_factor, cooking_factor_error, rank, chisq = lstsq(X, orbit_y.values)
    p, dp = cooking_factor - 1, cooking_factor_error
    print(dp)
    print(
        f"for {quadrupole_name} bba changed gradient by  {bba_gradient_change}"
        f"\nneed to adjust forecast by {float(p)*100:.4f} +/- {float(dp)*100:.4f} %"
    )
    assert rank == 1
    # todo: where does the cooking factor come from!
    # todo: inaccuracy of the model calculation ?
    assert cooking_factor == pytest.approx(1, rel=0.2)
    assert cooking_factor_error == pytest.approx(0, abs=1e-6)

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

    orbit_y_ref = -orbit * bba_gradient_change
    dy = orbit_y - orbit_y_ref
    mean_square_error = np.sum(dy ** 2)
    mean_absolute_error = np.sum(np.absolute(dy))
    # in average the prediction is allowed to be wrong by 10 microns for
    # each element position .. divide by element to get an average feeling

    n_elements = len(acc)
    assert mean_square_error / n_elements < 10e-6
    # needs roughly 10 times more
    assert mean_absolute_error / n_elements < 10 * 10e-6

    # scaling plot
    orbit_y_expected = -orbit * bba_gradient_change
    orbit_y_fit = -orbit * bba_gradient_change * cooking_factor

    dy_expected = orbit_y - orbit_y_expected
    dy_fit = orbit_y - orbit_y_fit

    pscale = 1000
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax, ax_diff = axes
    ax.plot(db.s, orbit_y * pscale, "-", label="closed orbit")
    # todo: negative sign for vertical plane ... where to handle it properly
    ax.plot(db.s, orbit_y_expected * pscale, "-", label="Forcast from twiss")
    ax.plot(
        db.s,
        orbit_y_fit * pscale,
        "-",
        label=f"Forcast from twiss scaled by {float(cooking_factor):.3f}",
    )
    ax.legend()
    ax.set_ylabel("y [mm]")
    ax_diff.plot(db.s, dy_expected * pscale, label="difference to twiss estimate")
    ax_diff.plot(
        db.s,
        dy_fit * pscale,
        label=f"difference to twiss estimate scaled by {float(cooking_factor):.3f}",
    )
    ax_diff.set_xlabel("s [m]")
    ax_diff.set_ylabel("y [mm]")
    ax.legend()
    plt.show()
