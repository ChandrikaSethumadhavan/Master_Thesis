import os
import pandas as pd
import numpy as np

import config
from metadata import EXPERIMENTS, K_VALUES
from helpers import get_pressure_col, ml_to_m3, mM_ml_to_umol
from battino import gamma_from_T_series


def build_summary_table():
    rows = []

    for (sample, sensor, temp_label, C_mM, Vsol_ml, Vg_ml, csv_file) in EXPERIMENTS:
        k = K_VALUES.get((sample, sensor), None)
        if k is None:
            raise ValueError(f"Missing k for {(sample, sensor)}")

        csv_path = os.path.join(config.CSV_FOLDER, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        P_col = get_pressure_col(temp_label, sensor)
        if P_col not in df.columns:
            raise KeyError(f"Column '{P_col}' not found in {csv_file}")

        t = df[config.TIME_COL].values.astype(float)
        t = t - t[0]

        P_meas = df[P_col].values.astype(float)
        P_meas = P_meas - P_meas[0]

        if temp_label == "RT":
            Tk = df[config.TEMP_COL_K].values.astype(float)
        else:
            Tk = df[config.TEMP_COL_37C].values.astype(float) + 273.15

        if np.mean(Tk) < 200:
            raise ValueError(f"Temperature unit error in {csv_file}")

        gamma_t = gamma_from_T_series(pd.Series(Tk), Vsol_ml, Vg_ml).values.astype(float)

        idx_peak = int(np.argmax(P_meas))
        P_peak_kPa = float(P_meas[idx_peak])
        T_peak = float(Tk[idx_peak])
        gamma_peak = float(gamma_t[idx_peak])

        P_peak_Pa = P_peak_kPa * 1000.0
        n_gas_peak_mol = (P_peak_Pa * ml_to_m3(Vg_ml)) / (config.R_J * T_peak)

        n_stoich_umol = mM_ml_to_umol(C_mM, Vsol_ml)
        n_gas_peak_umol = n_gas_peak_mol * 1e6

        eta_peak = n_gas_peak_umol / n_stoich_umol
        beta_peak = eta_peak * (1.0 + gamma_peak)

        rows.append({
            "sample": sample,
            "sample_prefix": sample.split("_")[0],
            "sensor": sensor,
            "temp_label": temp_label,
            "C_mM": float(C_mM),
            "Vsol_ml": float(Vsol_ml),
            "Vg_ml": float(Vg_ml),
            "k": float(k),
            "eta_peak": float(eta_peak),
            "beta_peak": float(beta_peak),
            "gamma_peak": float(gamma_peak),
            "csv": csv_file
        })

    df_sum = pd.DataFrame(rows)
    return df_sum