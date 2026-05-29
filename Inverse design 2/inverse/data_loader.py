import os
import numpy as np
import pandas as pd
import config
from metadata import EXPERIMENTS, K_VALUES
from helpers import get_pressure_col, ml_to_m3, mM_ml_to_mol
from battino import calc_H_SI

R = config.R_J


def calc_eta(P_peak_kPa, T_K, C_mM, Vsol, Vg):
    P = P_peak_kPa * 1000.0
    n_gas = P * ml_to_m3(Vg) / (R * T_K)
    n_aq  = calc_H_SI(T_K) * P * ml_to_m3(Vsol)
    return (n_gas + n_aq) / mM_ml_to_mol(C_mM, Vsol)


def build_summary_table():
    rows = []
    for (sample, sensor, temp_label, C_mM, Vsol, Vg, csv_file) in EXPERIMENTS:
        k = K_VALUES.get((sample, sensor))
        if k is None:
            raise ValueError(f"Missing k for {(sample, sensor)}")

        df = pd.read_csv(os.path.join(config.CSV_FOLDER, csv_file))
        P_col = get_pressure_col(temp_label, sensor)

        t      = df[config.TIME_COL].values.astype(float);  t -= t[0]
        P_meas = df[P_col].values.astype(float);            P_meas -= P_meas[0]

        Tk = df[config.TEMP_COL_K].values.astype(float)
        if temp_label == "37C":
            Tk += 273.15

        idx = int(np.argmax(P_meas))
        rows.append({
            "sample":        sample,
            "sample_prefix": sample.split("_")[0],
            "sensor":        sensor,
            "temp_label":    temp_label,
            "C_mM":          float(C_mM),
            "Vsol_ml":       float(Vsol),
            "Vg_ml":         float(Vg),
            "k":             float(k),
            "eta_peak":      calc_eta(float(P_meas[idx]), float(Tk[idx]), C_mM, Vsol, Vg),
            "csv":           csv_file,
        })
    return pd.DataFrame(rows)
