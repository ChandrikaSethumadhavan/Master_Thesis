import numpy as np
import config
from helpers import make_range
from parameter_estimation import estimate_eta_and_k
from physics_model import p_model_eta_max_kpa, time_to_threshold_h


def build_maps_for_temperature(df_sum, temp_c, sensor, C_mM):
    temp_label = config.MAP_TEMP_TO_LABEL[temp_c]
    eta, k, df_train = estimate_eta_and_k(df_sum, sensor, temp_label, C_mM)
    Tk        = temp_c + 273.15
    vsol_vals = make_range(config.VSOL_MIN_ML, config.VSOL_MAX_ML, config.VSOL_STEP_ML)
    P_map     = np.array([p_model_eta_max_kpa(Tk, eta, C_mM, v, config.VTOT_ML - v)
                          if config.VTOT_ML - v > 0 else 0.0 for v in vsol_vals])
    # t_map     = np.array([time_to_threshold_h(Tk, eta, C_mM, v, config.VTOT_ML - v, k, config.PACT_KPA)
    #                       if config.VTOT_ML - v > 0 else np.nan for v in vsol_vals])
    return {"temperature_c": temp_c, "sensor": sensor,
            "temp_label_used_for_training": temp_label, "C_target_mM": C_mM,
            "eta_hat": eta, "k_hat": k, "train_df": df_train,
            "vsol_vals": vsol_vals, "vg_vals": config.VTOT_ML - vsol_vals,
            "P_map": P_map, "t_act_map": t_map}


def build_all_design_maps(df_sum):
    return [build_maps_for_temperature(df_sum, T, config.MAP_SENSOR, config.MAP_CONCENTRATION_MM)
            for T in config.MAP_TEMPERATURES_C]
