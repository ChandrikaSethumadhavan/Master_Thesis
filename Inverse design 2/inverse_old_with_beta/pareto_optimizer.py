import numpy as np
import pandas as pd

import config
from helpers import make_range
from physics_model import p_model_beta_max_kpa, time_to_threshold_h


def compute_design_objectives(vsol_vals, vtot_vals, Tk, beta_hat, k_hat, C_mM):
    """
    Sweep (Vsol, Vtot) with Vg = Vtot - Vsol enforced and C fixed.

    Objectives (all minimise — genuinely conflicting):
      t_act   — time to reach actuation pressure (minimise → fast response)
      Vtot    — total chamber volume             (minimise → compact device)
      neg_Vg  — negative headspace = −Vg        (minimise → maximise Vg → safer/slower)

    Why these conflict:
      - Faster t_act needs high Vsol/Vg → small Vg → large neg_Vg is bad.
      - Smaller Vtot → smaller device → limits how large Vg can be.
      - Larger Vg (min neg_Vg) → needs larger Vtot → conflicts with compact goal.
    This forces three distinct corners: one with small Vg (fast), one with small
    Vtot (compact), and one with large Vg (pressure-safe / slow-release).
    """
    rows = []

    for vtot in vtot_vals:
        for vsol in vsol_vals:
            if vsol >= vtot:
                continue

            vg = vtot - vsol

            if vg < config.VG_MIN_ML:
                continue

            pmax = p_model_beta_max_kpa(Tk, beta_hat, C_mM, vsol, vg)

            if pmax < config.PACT_KPA:
                continue
            if pmax > config.P_SAFE_MAX_KPA:
                continue

            tact = time_to_threshold_h(
                Tk, beta_hat, C_mM, vsol, vg, k_hat, config.PACT_KPA
            )

            if np.isnan(tact):
                continue

            rows.append({
                "Vsol":   vsol,
                "Vg":     vg,
                "Vtot":   vtot,
                "Pmax":   pmax,
                "t_act":  tact,
                "neg_Vg": -vg,   # minimise this ↔ maximise Vg
            })

    return pd.DataFrame(rows)


def pareto_front(df):
    """Non-dominated front minimising [t_act, Vtot, neg_Vg]."""
    if df.empty:
        return df

    points    = df[["t_act", "Vtot", "neg_Vg"]].values
    is_pareto = np.ones(len(points), dtype=bool)

    for i, c in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(points[is_pareto] < c, axis=1)
            is_pareto[i] = True

    return df[is_pareto].reset_index(drop=True)


def extract_pareto_corners(df_pareto):
    """
    Three named extreme solutions from the Pareto front:

      min_tact — fastest actuation   (min t_act)
                 → high fill fraction, small Vg
      min_Vtot — most compact device (min Vtot)
                 → smallest total chamber that achieves Pact
      max_Vg   — largest headspace   (min neg_Vg = max Vg)
                 → pressure-safe / slow-release design with big gas buffer
    """
    return {
        "min_tact": df_pareto.loc[df_pareto["t_act"].idxmin()].to_dict(),
        "min_Vtot": df_pareto.loc[df_pareto["Vtot"].idxmin()].to_dict(),
        "max_Vg":   df_pareto.loc[df_pareto["neg_Vg"].idxmin()].to_dict(),  # min(neg_Vg) = max(Vg)
    }


def run_pareto_search(result):
    vsol_vals = result["vsol_vals"]
    vtot_vals = make_range(config.VTOT_MIN_ML, config.VTOT_MAX_ML, config.VTOT_STEP_ML)

    Tk       = result["temperature_c"] + 273.15
    beta_hat = result["beta_hat"]
    k_hat    = result["k_hat"]
    C        = result["C_target_mM"]

    df = compute_design_objectives(vsol_vals, vtot_vals, Tk, beta_hat, k_hat, C)

    if df.empty:
        return df, df, {}

    df_pareto = pareto_front(df)
    corners   = extract_pareto_corners(df_pareto)

    return df, df_pareto, corners
