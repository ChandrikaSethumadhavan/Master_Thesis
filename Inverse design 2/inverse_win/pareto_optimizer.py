import numpy as np
import pandas as pd

import config
from physics_model import p_model_beta_max_kpa, time_to_threshold_h


def compute_design_objectives(
        vg_vals,
        vsol_vals,
        Tk,
        beta_hat,
        k_hat,
        C_mM):

    rows = []

    for vsol in vsol_vals:
        for vg in vg_vals:

            pmax = p_model_beta_max_kpa(
                Tk,
                beta_hat,
                C_mM,
                vsol,
                vg
            )

            if pmax < config.PACT_KPA:
                continue

            tact = time_to_threshold_h(
                Tk,
                beta_hat,
                C_mM,
                vsol,
                vg,
                k_hat,
                config.PACT_KPA
            )

            if np.isnan(tact):
                continue

            rows.append({
                "Vsol": vsol,
                "Vg": vg,
                "Pmax": pmax,
                "t_act": tact
            })

    return pd.DataFrame(rows)

def pareto_front(df):

    points = df[["t_act", "Vsol", "Vg"]].values
    is_pareto = np.ones(points.shape[0], dtype=bool)

    for i, c in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(
                points[is_pareto] < c,
                axis=1
            )
            is_pareto[i] = True

    return df[is_pareto]

def run_pareto_search(result):

    vg_vals = result["vg_vals"]
    vsol_vals = result["vsol_vals"]

    Tk = result["temperature_c"] + 273.15
    beta_hat = result["beta_hat"]
    k_hat = result["k_hat"]
    C = result["C_target_mM"]

    df = compute_design_objectives(
        vg_vals,
        vsol_vals,
        Tk,
        beta_hat,
        k_hat,
        C
    )

    df_pareto = pareto_front(df)

    return df, df_pareto