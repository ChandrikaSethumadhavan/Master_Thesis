import numpy as np
import config


def ml_to_m3(ml):
    return ml * 1e-6


def mM_ml_to_umol(C_mM, V_ml):
    return C_mM * V_ml


def get_pressure_col(temp_label, sensor):
    if temp_label == "RT":
        return config.PRESS_COL_RT_ABP2 if sensor == "ABP2" else config.PRESS_COL_RT_MPR
    return config.PRESS_COL_37C_ABP2 if sensor == "ABP2" else config.PRESS_COL_37C_MPR


def interp_piecewise_clamped(x_train, y_train, x_query):
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if len(x) < 2:
        raise ValueError("Need at least 2 points for interpolation.")

    return float(np.interp(float(x_query), x, y))


def make_range(start, stop, step):
    n = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, n)