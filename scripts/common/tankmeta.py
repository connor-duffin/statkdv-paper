import logging

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix


logger = logging.getLogger(__name__)
DATA_FILE = "data/Processed/2-layer/ex030497/WAVE0.DAT"


def load_data():
    """Load wave data from file

    loads data from
        data/Processed/2-layer/ex030497/WAVE0.DAT

    and truncates to time < 301 s.
    """
    # wavea
    # t = 1.417, 29.716 gives
    def wavea_convert(volts):
        d = [-1.4379, -1.0343]  # displacement
        v = [1.46, 1.3428]  # voltages
        return(d[0] + (d[1] - d[0]) / (v[1] - v[0]) * (volts - v[0]))

    # waveb
    # t = 1.56, 40.748
    def waveb_convert(volts):
        d = [-0.01219, -0.28029]
        v = [-0.1758, -0.0977]
        return(d[0] + (d[1] - d[0]) / (v[1] - v[0]) * (volts - v[0]))

    # wavec
    # t = 1.463, 41.85
    def wavec_convert(volts):
        d = [1.0865, -0.3944]
        v = [-0.3369, 0.1563]
        return(d[0] + (d[1] - d[0]) / (v[1] - v[0]) * (volts - v[0]))

    wave = pd.read_csv(
        DATA_FILE,
        names=['time', 'wg01', 'wg02', 'wg03', 'hgb01'],
        skiprows=6,
        sep=r'\s+'
    )[:-1]
    wave.time = pd.to_numeric(wave.time)
    wave = wave[wave.time < 301.]

    # convert (volts -> cm)
    wave['wg01'] = wavea_convert(wave.wg01) / 100
    wave['wg02'] = waveb_convert(wave.wg02) / 100
    wave['wg03'] = wavec_convert(wave.wg03) / 100

    return(wave)


def interpolate_data(wave, n_interp=20):
    """Interpolate from a wave dataframe to a more refined grid.

    takes in data from `load_data()` and outputs the values we want
    """
    if any(wave.columns != ["time", "wg01", "wg02", "wg03", "hgb01"]):
        logger.error("Wave data is incomplete.")

    x_obs = np.array([1.47, 3.02, 4.57])

    # copy to keep data
    wave = wave.copy()
    wave.drop(["time", "hgb01"], axis=1, inplace=True)

    x_interp = np.linspace(x_obs[0], x_obs[-1], n_interp)
    f_interp = interp1d(x_obs, wave.values)
    wave_interp = f_interp(x_interp)
    return(x_interp, wave_interp)


def interpolate_data_time(wave, t_grid):
    """ Interpolate from observed grid to the solution grid.

    Naturally filters out a bit of the noise.
    """
    t_obs = wave.time.values
    wave_processed = pd.DataFrame(data=t_grid, columns=["time"])

    idx_right = np.searchsorted(wave.time, t_grid)
    idx_left = idx_right - 1

    def interpol(x):
        return(x[idx_left]
               + (x[idx_right] - x[idx_left]) / (t_obs[idx_right] - t_obs[idx_left])
               * (t_grid - t_obs[idx_left]))

    # interpolate to our observation grid
    for col in wave:
        wave_processed[col] = interpol(wave[col].values)

    return(wave_processed)


def add_solitons(u_grid, x_grid):
    nt, nx = u_grid.shape
    u_obs = np.zeros((nt, nx // 2 + 1))
    x_obs = x_grid[0:(nx // 2 + 1)]
    for i in range(nt):
        u_reflect = np.append(u_grid[i, (nx // 2):], u_grid[i, 0])
        u_obs[i, :] = u_grid[i, 0:(nx // 2 + 1)] + u_reflect[::-1]

    return(u_obs, x_obs)


def add_soliton_operator(x_grid):
    """ Assume that the grid is evenly spaced w/ even number of gridpoints
    """
    n = len(x_grid)
    if n % 2 != 0:
        logger.error("can't have an odd number of gridpoints")
        raise ValueError

    n_obs = n // 2 + 1
    H = np.eye(n_obs, n)
    H[0, 0] *= 2
    H[1:, (n_obs - 1):] += np.eye(n_obs - 1)[::-1]
    return(csr_matrix(H))


# metadata, parameters, and settings
dedalus_settings = {"nt": 10001, "nx": 1024,
                    "nt_spectral": 50001, "nx_spectral": 1024,
                    "t_start": 0, "t_end": 300,
                    "x_start": 0, "x_end": 12}
statkdv_settings = {"nt": 1001, "nx": 200,
                    "t_start": 0, "t_end": 300,
                    "x_start": 0, "x_end": 12, "n_ensemble": 2048}

h1 = 0.232  # upper layer
h2 = 0.058  # lower layer
H = h1 + h2  # 0.29
delta_rho = 20
rho = 999.97
g = 9.81

g_prime = (delta_rho / rho) * g
c_nought = np.sqrt(g_prime * h1 * h2 / H)
alpha = 3 / 2 * c_nought * (h1 - h2) / (h1 * h2)
beta = 1 / 6 * c_nought * h1 * h2
horn_damping = 1 / 2 * np.sqrt(1e-6 * c_nought / 2) * (h1 + h2) / (h1 * h2)
amplitude = 0.012

parameters = {"alpha": alpha,
              "beta": beta,
              "c": c_nought,
              "horn_damping": horn_damping,
              "nu": 0.003}

control = {"newton_iter": 50,
           "newton_tol": 1e-8,
           "thin": 1,
           "noisefree": True,
           "save_ensemble": False}

priors = {"scale_G_mean": 1, "scale_G_sd": 1,
          "length_G_mean": 1, "length_G_sd": 1,
          "sigma_y_mean": 0, "sigma_y_sd": 0.02}

statkdv_initial_condition = (
    f"Piecewise("
    f"(2 * {amplitude} / 6 * x - {amplitude}, x < 6),"
    f"(-2 * {amplitude} / 6 * x + 3 * {amplitude}, True))"
)
