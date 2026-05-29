import numpy as np
import config
from battino import calc_H_SI
from helpers import ml_to_m3, mM_ml_to_mol

R = config.R_J


def p_model_eta_time_kpa(t_h, Tk, eta, C_mM, Vsol, Vg, k):
    t_h = np.asarray(t_h, float)
    Tk  = np.asarray(Tk,  float)
    if Tk.ndim == 0 or Tk.size == 1:
        Tk = np.full_like(t_h, float(Tk))
    num = eta * mM_ml_to_mol(C_mM, Vsol) * (1.0 - np.exp(-k * t_h))
    den = ml_to_m3(Vg) / (R * Tk) + calc_H_SI(Tk) * ml_to_m3(Vsol)
    return (num / den) / 1000.0


def p_model_eta_max_kpa(Tk, eta, C_mM, Vsol, Vg):
    Tk = float(Tk)
    den = ml_to_m3(Vg) / (R * Tk) + float(calc_H_SI(Tk)) * ml_to_m3(Vsol)
    return (eta * mM_ml_to_mol(C_mM, Vsol) / den) / 1000.0


def time_to_threshold_h(Tk, eta, C_mM, Vsol, Vg, k, P_thresh):
    pmax = p_model_eta_max_kpa(Tk, eta, C_mM, Vsol, Vg)
    if pmax <= P_thresh or P_thresh <= 0:
        return np.nan
    frac = 1.0 - P_thresh / pmax
    return np.nan if frac <= 0 else -np.log(frac) / k
