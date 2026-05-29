import numpy as np
import pandas as pd
import config

# Battino et al. 1983
battino_T = np.array([
    273.15, 278.15, 283.15, 288.15, 293.15,
    298.15, 303.15, 308.15, 313.15, 318.15,
    323.15, 328.15, 333.15, 338.15, 343.15, 348.15
])

battino_dH = np.array([
    -17.60, -16.18, -15.16, -14.15, -13.13,
    -12.11, -11.10, -10.08, -9.06, -8.05,
    -7.03, -6.01, -5.00, -3.98, -2.97, -1.95
])

kH_ref = 1.3e-3
T_ref = 298.15
dH_ref = -12.11


def interpolate_dH(Tk):
    return np.interp(Tk, battino_T, battino_dH)


def calc_kH(Tk):
    dH_T = interpolate_dH(Tk)
    dH_avg = (dH_ref + dH_T) / 2.0 * 1000.0
    return kH_ref * np.exp(-(dH_avg / config.R_GAS) * (1.0 / Tk - 1.0 / T_ref))


def gamma_from_T_series(Tk_series, Vsol_ml, Vg_ml):
    kH = Tk_series.apply(calc_kH)
    return kH * config.R_ATM * Tk_series * (Vsol_ml / Vg_ml)


def gamma_scalar(Tk, Vsol_ml, Vg_ml):
    kH = calc_kH(Tk)
    return kH * config.R_ATM * Tk * (Vsol_ml / Vg_ml)