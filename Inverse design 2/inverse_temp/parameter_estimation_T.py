"""
Temperature-aware parameter estimation using Arrhenius scaling.

For k:
  k(T, C) = k_GPR_RT(C)  ×  exp( −Eₐ/R × (1/T − 1/T_RT) )

  Eₐ is fitted from samples that have experimental data at BOTH RT and 37 °C.
  k_GPR_RT(C) comes from the existing GPR trained on the RT group.

For β:
  β(T, C) = β_RT(C)  +  [β_37C(C) − β_RT(C)] × (T − T_RT) / (T_37C − T_RT)

  Linear interpolation between the two temperature groups.
  Mild extrapolation (frac slightly outside [0,1]) is allowed but clamped.

No changes to physics_model.py or battino.py are needed — they already
accept arbitrary T as a continuous input.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inverse'))

import numpy as np
import config                                    # loads inverse_temp/config.py
from helpers import interp_piecewise_clamped
from parameter_estimation import fit_k_gpr, predict_k_gpr   # base GPR helpers

R_GAS = 8.314   # J/(mol·K)


# ─────────────────────────────────────────────────────────────────────────────
# Arrhenius activation energy
# ─────────────────────────────────────────────────────────────────────────────

def fit_arrhenius_ea(df_sum, sensor):
    """
    Estimate mean activation energy Eₐ (J/mol) from samples that have
    measurements at BOTH RT and 37 °C.

    Formula:  Eₐ = −R · ln(k_37C / k_RT) / (1/T_37C − 1/T_RT)

    Returns: float Eₐ in J/mol.
    """
    df_train = df_sum[
        (df_sum["sample_prefix"] != config.TEST_SAMPLE_PREFIX) &
        (df_sum["sensor"]        == sensor)
    ].copy()

    rt_prefixes  = set(df_train[df_train["temp_label"] == "RT" ]["sample_prefix"])
    c37_prefixes = set(df_train[df_train["temp_label"] == "37C"]["sample_prefix"])
    paired       = rt_prefixes & c37_prefixes

    if len(paired) == 0:
        raise ValueError(
            f"No samples with data at BOTH RT and 37 °C for sensor={sensor}.\n"
            f"  RT  samples : {rt_prefixes}\n"
            f"  37C samples : {c37_prefixes}"
        )

    Ea_values = []
    for prefix in sorted(paired):
        row_rt  = df_train[(df_train["sample_prefix"] == prefix) &
                           (df_train["temp_label"]    == "RT" )].iloc[0]
        row_37c = df_train[(df_train["sample_prefix"] == prefix) &
                           (df_train["temp_label"]    == "37C")].iloc[0]

        k_rt  = float(row_rt["k"])
        k_37c = float(row_37c["k"])

        if k_rt <= 0 or k_37c <= 0:
            print(f"  [Arrhenius] Skipping {prefix}: non-positive k value.")
            continue

        Ea = (-R_GAS * np.log(k_37c / k_rt)
              / (1.0 / config.T_REF_37C_K - 1.0 / config.T_REF_RT_K))
        Ea_values.append(Ea)
        print(f"  [Arrhenius] {prefix} ({sensor}): "
              f"k_RT={k_rt:.5f}, k_37C={k_37c:.5f}, "
              f"Eₐ = {Ea/1000:.1f} kJ/mol")

    if len(Ea_values) == 0:
        raise ValueError("All paired samples had non-positive k values.")

    Ea_mean = float(np.mean(Ea_values))
    print(f"  [Arrhenius] Mean Eₐ ({sensor}) = {Ea_mean/1000:.1f} kJ/mol "
          f"from {len(Ea_values)} sample pair(s)")
    return Ea_mean


# ─────────────────────────────────────────────────────────────────────────────
# k at arbitrary temperature
# ─────────────────────────────────────────────────────────────────────────────

def predict_k_at_T(gpr_rt, Ea, C_mM, T_kelvin):
    """
    k(T, C) = k_GPR_RT(C) × exp(−Eₐ/R × (1/T − 1/T_RT))

    gpr_rt : fitted GaussianProcessRegressor from the RT training group.
    Ea     : activation energy in J/mol from fit_arrhenius_ea().
    """
    k_rt   = predict_k_gpr(gpr_rt, C_mM)
    scale  = np.exp(-Ea / R_GAS * (1.0 / T_kelvin - 1.0 / config.T_REF_RT_K))
    return max(float(k_rt * scale), 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# β at arbitrary temperature
# ─────────────────────────────────────────────────────────────────────────────

def predict_beta_at_T(df_sum, sensor, C_mM, T_kelvin):
    """
    β(T, C) = β_RT(C) + [β_37C(C) − β_RT(C)] × (T − T_RT)/(T_37C − T_RT)

    Linear interpolation (mild extrapolation clamped to ×2 of range).
    """
    df_train = df_sum[
        (df_sum["sample_prefix"] != config.TEST_SAMPLE_PREFIX) &
        (df_sum["sensor"]        == sensor)
    ].copy()

    df_rt  = df_train[df_train["temp_label"] == "RT" ]
    df_37c = df_train[df_train["temp_label"] == "37C"]

    beta_rt  = interp_piecewise_clamped(
        df_rt["C_mM"].values,  df_rt["beta_peak"].values,  C_mM)
    beta_37c = interp_piecewise_clamped(
        df_37c["C_mM"].values, df_37c["beta_peak"].values, C_mM)

    frac = ((T_kelvin - config.T_REF_RT_K) /
            (config.T_REF_37C_K - config.T_REF_RT_K))
    frac = float(np.clip(frac, -0.5, 1.5))   # allow mild extrapolation

    return float(beta_rt + (beta_37c - beta_rt) * frac)
