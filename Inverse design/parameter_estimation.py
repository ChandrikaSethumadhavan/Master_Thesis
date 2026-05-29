import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import config
from helpers import interp_piecewise_clamped


def fit_k_gpr(df_train):
    X = df_train[["C_mM"]].values.astype(float)
    y = df_train["k"].values.astype(float)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-1))
    )

    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gpr.fit(X, y)
    return gpr


def predict_k_gpr(gpr, C_mM):
    pred = gpr.predict(np.array([[float(C_mM)]]), return_std=False)[0]
    return max(float(pred), 1e-6)


def estimate_beta_and_k(df_sum, sensor, temp_label, C_target_mM):
    df_train = df_sum[
        (df_sum["sample_prefix"] != config.TEST_SAMPLE_PREFIX) &
        (df_sum["sensor"] == sensor) &
        (df_sum["temp_label"] == temp_label)
    ].copy()

    if len(df_train) < 2:
        raise ValueError(
            f"Not enough training points for sensor={sensor}, temp_label={temp_label}"
        )

    beta_hat = interp_piecewise_clamped(df_train["C_mM"], df_train["beta_peak"], C_target_mM)

    gpr = fit_k_gpr(df_train)
    k_hat = predict_k_gpr(gpr, C_target_mM)

    return beta_hat, k_hat, df_train