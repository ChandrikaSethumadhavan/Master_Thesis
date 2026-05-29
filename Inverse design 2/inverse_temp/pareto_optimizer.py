"""
Temperature-aware Pareto optimiser.

Design variables : Vsol, Vg  — swept INDEPENDENTLY (no Vtot constraint)
Fixed            : C = MAP_CONCENTRATION_MM
Swept            : T ∈ [T_MIN_C, T_MAX_C]

Why (Vsol, Vg) and not (Vsol, Vtot)?
  Vsol and Vg are the direct physical inputs to the pressure model:
      P  ∝  C × Vsol / Vg
  They are what you actually pipette and machine in the lab.
  Vtot = Vsol + Vg is a derived quantity with no independent role in
  the physics.  Now that temperature provides a third axis of diversity,
  there is no need for the artificial Vtot coupling used in inverse/.

Objectives (all minimise — genuinely conflicting):
  t_act     — time to actuation (h)
              Favours: high T (faster kinetics) + large Vsol + small Vg.
  T_celsius — operating temperature (°C)
              Favours: 25 °C RT designs.
              Conflicts with t_act: lower T → slower kinetics → longer wait.
  Vsol      — solution volume (mL) = proxy for reagent amount used
              Favours: small fill.
              Conflicts with t_act: less reagent → less O2 → slower pressure rise.

Three corner solutions:
  min_tact  — fastest actuator  (hot, high Vsol, small Vg)
  min_T     — coolest design    (RT, geometry compensates)
  min_Vsol  — least reagent     (small fill, higher T or smaller Vg compensates)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inverse'))

import numpy as np
import pandas as pd

import config
from helpers import make_range
from physics_model import p_model_beta_max_kpa, time_to_threshold_h
from parameter_estimation import fit_k_gpr
from parameter_estimation_T import (
    fit_arrhenius_ea, predict_k_at_T, predict_beta_at_T
)


def compute_design_objectives(vsol_vals, vg_vals, T_vals_K,
                               C_mM, df_sum, sensor, Ea, gpr_rt):
    """
    Sweep every (Vsol, Vg, T) combination independently.
    No Vtot constraint — Vsol and Vg are free variables.
    """
    rows = []

    for T_K in T_vals_K:
        T_c      = T_K - 273.15
        beta_hat = predict_beta_at_T(df_sum, sensor, C_mM, T_K)
        k_hat    = predict_k_at_T(gpr_rt, Ea, C_mM, T_K)

        for vg in vg_vals:
            for vsol in vsol_vals:

                pmax = p_model_beta_max_kpa(T_K, beta_hat, C_mM, vsol, vg)
                if pmax < config.PACT_KPA or pmax > config.P_SAFE_MAX_KPA:
                    continue

                tact = time_to_threshold_h(
                    T_K, beta_hat, C_mM, vsol, vg, k_hat, config.PACT_KPA
                )
                if np.isnan(tact):
                    continue

                rows.append({
                    "Vsol":  vsol,
                    "Vg":    vg,
                    "T_C":   T_c,
                    "Pmax":  pmax,
                    "t_act": tact,
                })

    return pd.DataFrame(rows)


def pareto_front(df):
    """Non-dominated front minimising [t_act, T_C, Vsol]."""
    if df.empty:
        return df

    points    = df[["t_act", "T_C", "Vsol"]].values
    is_pareto = np.ones(len(points), dtype=bool)

    for i, c in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(points[is_pareto] < c, axis=1)
            is_pareto[i] = True

    return df[is_pareto].reset_index(drop=True)


def extract_pareto_corners(df_pareto):
    return {
        "min_tact": df_pareto.loc[df_pareto["t_act"].idxmin()].to_dict(),
        "min_T":    df_pareto.loc[df_pareto["T_C"].idxmin()].to_dict(),
        "min_Vsol": df_pareto.loc[df_pareto["Vsol"].idxmin()].to_dict(),
    }


def run_pareto_search(df_sum, sensor, C_mM):
    vsol_vals = make_range(config.VSOL_MIN_ML, config.VSOL_MAX_ML,
                           config.VSOL_STEP_ML)
    vg_vals   = make_range(config.VG_MIN_ML,  config.VG_MAX_ML,
                           config.VG_STEP_ML)
    T_vals_K  = make_range(config.T_MIN_C, config.T_MAX_C,
                           config.T_STEP_C) + 273.15

    print(f"\n  Fitting Arrhenius Eₐ from paired RT/37 °C data...")
    Ea = fit_arrhenius_ea(df_sum, sensor)

    df_rt = df_sum[
        (df_sum["sample_prefix"] != config.TEST_SAMPLE_PREFIX) &
        (df_sum["sensor"]        == sensor) &
        (df_sum["temp_label"]    == "RT")
    ].copy()
    gpr_rt = fit_k_gpr(df_rt)

    n_eval = len(vsol_vals) * len(vg_vals) * len(T_vals_K)
    print(f"  Sweeping {len(vsol_vals)} Vsol × {len(vg_vals)} Vg "
          f"× {len(T_vals_K)} T = {n_eval:,} evaluations...")

    df = compute_design_objectives(
        vsol_vals, vg_vals, T_vals_K, C_mM, df_sum, sensor, Ea, gpr_rt
    )

    if df.empty:
        return df, df, {}

    df_pareto = pareto_front(df)
    corners   = extract_pareto_corners(df_pareto)

    return df, df_pareto, corners
