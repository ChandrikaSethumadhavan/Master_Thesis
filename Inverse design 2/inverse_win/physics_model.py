import numpy as np
import config
from helpers import ml_to_m3, mM_ml_to_umol
from battino import gamma_scalar


def p_model_beta_time_kpa(t_h, Tk, beta_hat, C_mM, Vsol_ml, Vg_ml, k):
    t_h = np.asarray(t_h, dtype=float)
    Tk = np.asarray(Tk, dtype=float)

    if Tk.size == 1:
        Tk = np.full_like(t_h, float(Tk), dtype=float)

    gamma_t = np.array([gamma_scalar(temp, Vsol_ml, Vg_ml) for temp in Tk], dtype=float)

    n_stoich_umol = mM_ml_to_umol(C_mM, Vsol_ml)
    n_total_umol = beta_hat * n_stoich_umol * (1.0 - np.exp(-k * t_h))
    n_gas_umol = n_total_umol / (1.0 + gamma_t)
    n_gas_mol = n_gas_umol * 1e-6
    P_Pa = (n_gas_mol * config.R_J * Tk) / ml_to_m3(Vg_ml)
    return P_Pa / 1000.0


def p_model_beta_max_kpa(Tk, beta_hat, C_mM, Vsol_ml, Vg_ml):
    gamma_t = gamma_scalar(Tk, Vsol_ml, Vg_ml)

    n_stoich_umol = mM_ml_to_umol(C_mM, Vsol_ml)
    n_total_umol = beta_hat * n_stoich_umol
    n_gas_umol = n_total_umol / (1.0 + gamma_t)
    n_gas_mol = n_gas_umol * 1e-6
    P_Pa = (n_gas_mol * config.R_J * Tk) / ml_to_m3(Vg_ml)
    return P_Pa / 1000.0


def time_to_threshold_h(Tk, beta_hat, C_mM, Vsol_ml, Vg_ml, k, P_threshold_kPa):
    pmax = p_model_beta_max_kpa(Tk, beta_hat, C_mM, Vsol_ml, Vg_ml)

    if P_threshold_kPa <= 0:
        return np.nan
    if pmax <= P_threshold_kPa:
        return np.nan

    frac = 1.0 - (P_threshold_kPa / pmax)
    if frac <= 0:
        return np.nan

    t_h = -np.log(frac) / k
    return t_h