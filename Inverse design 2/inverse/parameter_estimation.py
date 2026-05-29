import numpy as np
import config
from helpers import interp_piecewise_clamped


def estimate_eta_and_k(df_sum, sensor, temp_label, C_mM):
    df = df_sum[
        (df_sum["sample_prefix"] != config.TEST_SAMPLE_PREFIX) &
        (df_sum["sensor"]        == sensor) &
        (df_sum["temp_label"]    == temp_label)
    ].copy()
    if len(df) < 2:
        raise ValueError(f"Need ≥2 training points for sensor={sensor}, temp={temp_label}")
    return (
        interp_piecewise_clamped(df["C_mM"], df["eta_peak"], C_mM),
        float(np.mean(df["k"])),
        df,
    )
