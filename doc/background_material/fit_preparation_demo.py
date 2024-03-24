import numpy as np
from scipy.linalg import lstsq

n_exc = 3
n_orb = 5


def demonstrate_1d():
    # prepare matrix ready for fit of bpm offsets and scale of
    # orbit at once
    #: n_orb ... number of orbit positions
    #: n_exc ... number of excitations
    orbit = np.arange(1, n_orb + 1)
    excitations = 10 * np.arange(1, n_exc + 1)

    A = np.zeros([n_exc, n_orb, n_orb + 1], dtype=float)
    idx = np.arange(n_orb)
    # for fitting bpm positions
    A[:, idx, idx] = 1

    # orbit times excitation ... to set on A
    t_orb = excitations[:, np.newaxis] * orbit[np.newaxis, :]
    A[:, :, -1] = t_orb

    Ar = A.reshape(-1, n_orb+1)

    meas_tmp = orbit * 1.5
    bpm_offset = [0.2, -0.4, 0.6, -0.8, 0.7]
    measurements = np.array(excitations[:, np.newaxis] * meas_tmp[np.newaxis, :], dtype=float)
    idxm = np.arange(measurements.shape[0])
    measurements[idxm, :] += bpm_offset
    b = measurements.ravel()
    p, residues, rank, s = lstsq(Ar, b)
    p

def demonstrate_2d():

    orbit_tmp = np.arange(1, 5 + 1)
    orbit_x = -1 * orbit_tmp
    orbit_y = 1 * orbit_tmp[2::2]
    excitations = 10 * np.arange(1, n_exc + 1)

    n_orb_x = len(orbit_x)
    n_orb_y = len(orbit_y)
    n_orb = n_orb_x + n_orb_y

    A = np.zeros([n_exc, n_orb, n_orb + 1], dtype=float)
    # for bpm offsets
    idx = np.arange(n_orb)

    # for fitting bpm positions
    A[:, idx, idx] = 1

    # prepare orbits to fill
    t_orb_x = excitations[:, np.newaxis] * orbit_x[np.newaxis, :]
    t_orb_y = excitations[:, np.newaxis] * orbit_y[np.newaxis, :]
    A[:, :n_orb_x, -1] = t_orb_x
    A[:, n_orb_x:, -1] = t_orb_y
    Ar = A.reshape(-1, n_orb + 1)
    Ar

    measurements_x = orbit_x * 2.3
    measurements_y = orbit_y * 2.3
    meas_tmp_x = excitations[:, np.newaxis] * measurements_x[np.newaxis, :]
    meas_tmp_y = excitations[:, np.newaxis] * measurements_y[np.newaxis, :]

    b = np.hstack([
            meas_tmp_x, meas_tmp_y
    ])
    br = b.ravel()
    p, residues, rank, s = lstsq(Ar, br)
    bpm_x_pos, bpm_y_pos = p[:n_orb_x], p[n_orb_x:]
    p

if __name__ == "__main__":
    demonstrate_1d()

if __name__ == "__main__":
    demonstrate_2d()
