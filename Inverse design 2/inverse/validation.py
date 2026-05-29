import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from helpers import get_pressure_col
from parameter_estimation import estimate_eta_and_k
from physics_model import p_model_eta_time_kpa, p_model_eta_max_kpa


def validate_m6_curves(df_sum, sensor="ABP2"):
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    df_test = df_sum[(df_sum["sample_prefix"] == config.TEST_SAMPLE_PREFIX) & (df_sum["sensor"] == sensor)].copy()
    if len(df_test) == 0:
        raise ValueError(f"No rows for {config.TEST_SAMPLE_PREFIX} sensor={sensor}")

    results, payloads = [], []
    for _, tr in df_test.iterrows():
        sample, temp_label = tr["sample"], tr["temp_label"]
        C, Vsol, Vg = float(tr["C_mM"]), float(tr["Vsol_ml"]), float(tr["Vg_ml"])

        eta, k, df_train = estimate_eta_and_k(df_sum, sensor, temp_label, C)

        df    = pd.read_csv(os.path.join(config.CSV_FOLDER, tr["csv"]))
        P_col = get_pressure_col(temp_label, sensor)
        t     = df[config.TIME_COL].values.astype(float);  t -= t[0]
        P_m   = df[P_col].values.astype(float);            P_m -= P_m[0]

        T_nom = 298.15 if temp_label == "RT" else 310.15
        Tk    = np.full(len(t), T_nom)
        P_p   = p_model_eta_time_kpa(t, Tk, eta, C, Vsol, Vg, k)
        Pmax  = p_model_eta_max_kpa(T_nom, eta, C, Vsol, Vg)
        rmse  = float(np.sqrt(np.mean((P_m - P_p) ** 2)))

        results.append({"sample": sample, "temp_label": temp_label, "C_mM": C,
                        "eta_hat": round(eta, 6), "k_hat_mean": round(k, 6),
                        "rmse_kpa": round(rmse, 4), "pmax_meas_kpa": round(float(np.max(P_m)), 3),
                        "pmax_pred_kpa": round(Pmax, 3)})
        payloads.append({"sample": sample, "temp_label": temp_label, "t_h": t, "P_meas": P_m, "P_pred": P_p})

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, P_m, "k-",  lw=2,   label="Measured (M6)")
        ax.plot(t, P_p, "b--", lw=1.8, label=f"Predicted  η={eta:.4f}  k_mean={k:.5f} h⁻¹  RMSE={rmse:.3f} kPa")
        ax.axhline(Pmax,              color="grey", ls=":",  lw=1.4, label=f"Eq17 ΔP_max={Pmax:.1f} kPa")
        ax.axhline(config.PACT_KPA,   color="red",  ls="--", lw=1.2, alpha=0.6, label=f"P_act={config.PACT_KPA} kPa")
        ax.set_xlabel("Time (h)"); ax.set_ylabel("ΔP (kPa)")
        ax.set_title(f"M6 Validation  |  {sample}  |  {sensor}  |  {temp_label}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_FOLDER, f"validation_{sample}_{sensor}.png"), dpi=150)
        plt.close()

    df_val = pd.DataFrame(results)
    df_val.to_csv(os.path.join(config.OUTPUT_FOLDER, f"validation_summary_{sensor}.csv"), index=False)
    return df_val, payloads
