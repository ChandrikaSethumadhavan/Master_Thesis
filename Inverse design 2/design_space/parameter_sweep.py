"""
Parameter exploration sweep: {C0, Vsol, Vg, T}
================================================
For each temperature, estimates eta and k from training data,
then sweeps (Vsol, Vg) independently to build P_max grids.
"""

import numpy as np
import config
from helpers import make_range
from parameter_estimation import estimate_eta_and_k
from physics_model import p_model_eta_max_kpa, time_to_threshold_h
from design_constraint import build_A_grid


def sweep_temperature(df_sum, temp_c: float, sensor: str, C_mM: float) -> dict:
    """Run the full parameter sweep for one temperature.

    Returns a result dict with P_max, t_act, A, f_gas grids and metadata.
    """
    temp_label = config.TEMP_LABEL_MAP[temp_c]
    Tk         = config.TEMP_TK_MAP[temp_c]

    eta, k, df_train = estimate_eta_and_k(df_sum, sensor, temp_label, C_mM)

    vsol_vals = make_range(config.VSOL_MIN_ML, config.VSOL_MAX_ML, config.VSOL_STEP_ML)
    vg_vals   = make_range(config.VG_MIN_ML,   config.VG_MAX_ML,   config.VG_STEP_ML)

    P_grid = np.full((len(vg_vals), len(vsol_vals)), np.nan)
    T_grid = np.full((len(vg_vals), len(vsol_vals)), np.nan)

    for i, vg in enumerate(vg_vals):
        for j, vs in enumerate(vsol_vals):
            P_grid[i, j] = p_model_eta_max_kpa(Tk, eta, C_mM, vs, vg)
            if P_grid[i, j] >= config.PACT_KPA:
                T_grid[i, j] = time_to_threshold_h(Tk, eta, C_mM, vs, vg, k, config.PACT_KPA)

    A_grid, fgas_grid = build_A_grid(vsol_vals, vg_vals, Tk, eta)

    return {
        "temperature_c":  temp_c,
        "temp_label":     temp_label,
        "Tk":             Tk,
        "sensor":         sensor,
        "C_target_mM":   C_mM,
        "eta_hat":        eta,
        "k_hat":          k,
        "train_df":       df_train,
        "vsol_vals":      vsol_vals,
        "vg_vals":        vg_vals,
        "P_grid":         P_grid,    # P_max (kPa) — shape (Nvg, Nvsol)
        "T_grid":         T_grid,    # t_act (h)   — NaN where P_max < P_act
        "A_grid":         A_grid,    # design constraint A
        "fgas_grid":      fgas_grid, # gas fraction
    }


def run_all_sweeps(df_sum) -> list[dict]:
    return [
        sweep_temperature(df_sum, T, config.MAP_SENSOR, config.MAP_CONCENTRATION_MM)
        for T in config.MAP_TEMPERATURES_C
    ]
