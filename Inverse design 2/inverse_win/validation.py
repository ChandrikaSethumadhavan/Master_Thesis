import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from helpers import get_pressure_col
from parameter_estimation import estimate_beta_and_k
from physics_model import p_model_beta_time_kpa, p_model_beta_max_kpa


def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def first_crossing_time(time_h, signal_kpa, threshold_kpa):
    time_h = np.asarray(time_h, dtype=float)
    signal_kpa = np.asarray(signal_kpa, dtype=float)

    idx = np.where(signal_kpa >= threshold_kpa)[0]
    if len(idx) == 0:
        return np.nan
    return float(time_h[idx[0]])


def validate_m6_curves(df_sum, sensor="ABP2"):
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    df_test = df_sum[
        (df_sum["sample_prefix"] == config.TEST_SAMPLE_PREFIX) &
        (df_sum["sensor"] == sensor)
    ].copy()

    if len(df_test) == 0:
        raise ValueError(f"No held-out rows found for {config.TEST_SAMPLE_PREFIX} and sensor={sensor}")

    results = []
    curve_payloads = []

    for _, tr in df_test.iterrows():
        sample = tr["sample"]
        temp_label = tr["temp_label"]
        C_test = float(tr["C_mM"])
        Vsol_ml = float(tr["Vsol_ml"])
        Vg_ml = float(tr["Vg_ml"])
        csv_file = tr["csv"]

        beta_hat, k_hat, df_train = estimate_beta_and_k(
            df_sum=df_sum,
            sensor=sensor,
            temp_label=temp_label,
            C_target_mM=C_test
        )

        csv_path = os.path.join(config.CSV_FOLDER, csv_file)
        df = pd.read_csv(csv_path)

        P_col = get_pressure_col(temp_label, sensor)

        t_h = df[config.TIME_COL].values.astype(float)
        t_h = t_h - t_h[0]

        P_meas_kpa = df[P_col].values.astype(float)
        P_meas_kpa = P_meas_kpa - P_meas_kpa[0]

        if temp_label == "RT":
            Tk = df[config.TEMP_COL_K].values.astype(float)
        else:
            Tk = df[config.TEMP_COL_37C].values.astype(float) + 273.15

        if np.mean(Tk) < 200:
            raise ValueError(f"Temperature unit issue detected in {csv_file}")

        P_pred_kpa = p_model_beta_time_kpa(
            t_h=t_h,
            Tk=Tk,
            beta_hat=beta_hat,
            C_mM=C_test,
            Vsol_ml=Vsol_ml,
            Vg_ml=Vg_ml,
            k=k_hat
        )

        rmse = compute_rmse(P_meas_kpa, P_pred_kpa)
        mae = compute_mae(P_meas_kpa, P_pred_kpa)

        pmax_meas = float(np.max(P_meas_kpa))
        pmax_pred_curve = float(np.max(P_pred_kpa))

        pmax_pred_model = p_model_beta_max_kpa(
            Tk=float(np.mean(Tk)),
            beta_hat=beta_hat,
            C_mM=C_test,
            Vsol_ml=Vsol_ml,
            Vg_ml=Vg_ml
        )

        peak_error_kpa = pmax_pred_curve - pmax_meas
        peak_error_pct = 100.0 * peak_error_kpa / pmax_meas if pmax_meas > 0 else np.nan

        t_act_meas = first_crossing_time(t_h, P_meas_kpa, config.PACT_KPA)
        t_act_pred = first_crossing_time(t_h, P_pred_kpa, config.PACT_KPA)
        t_act_error_h = t_act_pred - t_act_meas if np.isfinite(t_act_meas) and np.isfinite(t_act_pred) else np.nan

        results.append({
            "sample": sample,
            "sensor": sensor,
            "temp_label": temp_label,
            "C_mM": C_test,
            "Vsol_ml": Vsol_ml,
            "Vg_ml": Vg_ml,
            "beta_hat": beta_hat,
            "k_hat_gpr_1_h": k_hat,
            "rmse_kpa": rmse,
            "mae_kpa": mae,
            "pmax_meas_kpa": pmax_meas,
            "pmax_pred_curve_kpa": pmax_pred_curve,
            "pmax_pred_model_kpa": pmax_pred_model,
            "peak_error_kpa": peak_error_kpa,
            "peak_error_pct": peak_error_pct,
            "t_act_meas_h": t_act_meas,
            "t_act_pred_h": t_act_pred,
            "t_act_error_h": t_act_error_h,
        })

        curve_payloads.append({
            "sample": sample,
            "sensor": sensor,
            "temp_label": temp_label,
            "t_h": t_h,
            "P_meas_kpa": P_meas_kpa,
            "P_pred_kpa": P_pred_kpa,
            "threshold_kpa": config.PACT_KPA,
        })

        plt.figure(figsize=(9, 5))
        plt.plot(t_h, P_meas_kpa, label="Measured M6")
        plt.plot(t_h, P_pred_kpa, "--", label=f"Predicted M6 (beta + GPR k={k_hat:.4f})")
        plt.axhline(config.PACT_KPA, linestyle=":", label=f"Threshold = {config.PACT_KPA:.1f} kPa")
        plt.xlabel("Time (h)")
        plt.ylabel("Pressure rise (kPa)")
        plt.title(f"M6 Validation | {sample} | Sensor={sensor}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.OUTPUT_FOLDER, f"validation_{sample}_{sensor}.png"),
            dpi=300
        )
        plt.close()

    df_val = pd.DataFrame(results)
    df_val.to_csv(os.path.join(config.OUTPUT_FOLDER, f"validation_summary_{sensor}.csv"), index=False)
    return df_val, curve_payloads