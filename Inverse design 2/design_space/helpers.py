import numpy as np
import config


def ml_to_m3(ml):       return ml * 1e-6
def mM_ml_to_mol(C, V): return C * V * 1e-6
def mM_ml_to_umol(C, V): return C * V


def get_pressure_col(temp_label, sensor):
    if temp_label == "RT":
        return config.PRESS_COL_RT_ABP2 if sensor == "ABP2" else config.PRESS_COL_RT_MPR
    return config.PRESS_COL_37C_ABP2 if sensor == "ABP2" else config.PRESS_COL_37C_MPR


def interp_piecewise_clamped(x_train, y_train, x_query):
    x, y = np.asarray(x_train, float), np.asarray(y_train, float)
    order = np.argsort(x)
    return float(np.interp(float(x_query), x[order], y[order]))


def make_range(start, stop, step):
    return np.linspace(start, stop, int(round((stop - start) / step)) + 1)
