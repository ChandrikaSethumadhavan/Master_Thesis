import numpy as np
import pandas as pd

import config
from helpers import make_range
from parameter_estimation import estimate_beta_and_k
from physics_model import p_model_beta_max_kpa, time_to_threshold_h


def build_maps_for_temperature(df_sum, map_temp_c, sensor, C_target_mM):

    temp_label = config.MAP_TEMP_TO_LABEL[map_temp_c]

    beta_hat, k_hat, df_train = estimate_beta_and_k(
        df_sum=df_sum,
        sensor=sensor,
        temp_label=temp_label,
        C_target_mM=C_target_mM
    )

    Tk = map_temp_c + 273.15

    vsol_vals = make_range(config.VSOL_MIN_ML, config.VSOL_MAX_ML, config.VSOL_STEP_ML)

    P_map = np.zeros(len(vsol_vals))
    feasible_map = np.zeros(len(vsol_vals))
    t_act_map = np.full(len(vsol_vals), np.nan)

    Vtot = config.VTOT_ML

    for i, Vsol_ml in enumerate(vsol_vals):

        Vg_ml = Vtot - Vsol_ml

        if Vg_ml <= 0:
            continue
        
        # fill_fraction = Vsol_ml / Vtot
        # if fill_fraction > config.FILL_FRACTION_MAX:
        #     continue

        Pmax_kPa = p_model_beta_max_kpa(
            Tk=Tk,
            beta_hat=beta_hat,
            C_mM=C_target_mM,
            Vsol_ml=Vsol_ml,
            Vg_ml=Vg_ml
        )

        P_map[i] = Pmax_kPa
        feasible_map[i] = 1 if Pmax_kPa >= config.PACT_KPA else 0

        t_act = time_to_threshold_h(
            Tk=Tk,
            beta_hat=beta_hat,
            C_mM=C_target_mM,
            Vsol_ml=Vsol_ml,
            Vg_ml=Vg_ml,
            k=k_hat,
            P_threshold_kPa=config.PACT_KPA
        )

        t_act_map[i] = t_act

    return {
        "temperature_c": map_temp_c,
        "sensor": sensor,
        "temp_label_used_for_training": temp_label,
        "C_target_mM": C_target_mM,
        "beta_hat": beta_hat,
        "k_hat": k_hat,
        "train_df": df_train,
        "vg_vals": config.VTOT_ML - vsol_vals,
        "vsol_vals": vsol_vals,
        "P_map": P_map,
        "feasible_map": feasible_map,
        "t_act_map": t_act_map,
    }


def build_all_design_maps(df_sum):
    results = []
    for temp_c in config.MAP_TEMPERATURES_C:
        result = build_maps_for_temperature(
            df_sum=df_sum,
            map_temp_c=temp_c,
            sensor=config.MAP_SENSOR,
            C_target_mM=config.MAP_CONCENTRATION_MM
        )
        results.append(result)
    return results