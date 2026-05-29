# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import config
# from physics_model import p_model_beta_max_kpa, time_to_threshold_h
# from parameter_estimation import estimate_beta_and_k


# def plot_validation_parity(df_val):
#     if len(df_val) == 0:
#         return

#     # Peak pressure parity
#     x = df_val["pmax_meas_kpa"].values
#     y = df_val["pmax_pred_curve_kpa"].values

#     mn = min(np.min(x), np.min(y))
#     mx = max(np.max(x), np.max(y))

#     plt.figure(figsize=(6, 6))
#     plt.scatter(x, y)
#     plt.plot([mn, mx], [mn, mx], "--")
#     for _, row in df_val.iterrows():
#         plt.annotate(row["sample"], (row["pmax_meas_kpa"], row["pmax_pred_curve_kpa"]))
#     plt.xlabel("Measured peak pressure (kPa)")
#     plt.ylabel("Predicted peak pressure (kPa)")
#     plt.title("Validation parity plot: peak pressure")
#     plt.tight_layout()
#     plt.savefig(os.path.join(config.OUTPUT_FOLDER, "validation_parity_peak_pressure.png"), dpi=300)
#     plt.close()

#     # Threshold time parity
#     df_t = df_val.dropna(subset=["t_act_meas_h", "t_act_pred_h"]).copy()
#     if len(df_t) > 0:
#         x = df_t["t_act_meas_h"].values
#         y = df_t["t_act_pred_h"].values
#         mn = min(np.min(x), np.min(y))
#         mx = max(np.max(x), np.max(y))

#         plt.figure(figsize=(6, 6))
#         plt.scatter(x, y)
#         plt.plot([mn, mx], [mn, mx], "--")
#         for _, row in df_t.iterrows():
#             plt.annotate(row["sample"], (row["t_act_meas_h"], row["t_act_pred_h"]))
#         plt.xlabel("Measured threshold time (h)")
#         plt.ylabel("Predicted threshold time (h)")
#         plt.title("Validation parity plot: threshold time")
#         plt.tight_layout()
#         plt.savefig(os.path.join(config.OUTPUT_FOLDER, "validation_parity_threshold_time.png"), dpi=300)
#         plt.close()


# def plot_m6_panel(curve_payloads):
#     if len(curve_payloads) == 0:
#         return

#     n = len(curve_payloads)
#     fig, axes = plt.subplots(n, 1, figsize=(9, 4 * n), squeeze=False)

#     for ax, payload in zip(axes[:, 0], curve_payloads):
#         ax.plot(payload["t_h"], payload["P_meas_kpa"], label="Measured")
#         ax.plot(payload["t_h"], payload["P_pred_kpa"], "--", label="Predicted")
#         ax.axhline(payload["threshold_kpa"], linestyle=":", label="Threshold")
#         ax.set_title(f"{payload['sample']} | {payload['sensor']} | {payload['temp_label']}")
#         ax.set_xlabel("Time (h)")
#         ax.set_ylabel("Pressure rise (kPa)")
#         ax.legend()

#     fig.tight_layout()
#     fig.savefig(os.path.join(config.OUTPUT_FOLDER, "m6_validation_panel.png"), dpi=300)
#     plt.close(fig)


# def plot_safe_operating_window(result):
#     X, Y = np.meshgrid(result["vg_vals"], result["vsol_vals"])
#     P = result["P_map"]

#     zone = np.zeros_like(P, dtype=int)
#     zone[P < config.PACT_KPA] = 0
#     zone[(P >= config.PACT_KPA) & (P <= config.P_SAFE_MAX_KPA)] = 1
#     zone[P > config.P_SAFE_MAX_KPA] = 2

#     plt.figure(figsize=(8, 6))
#     plt.contourf(X, Y, zone, levels=[-0.5, 0.5, 1.5, 2.5])
#     c1 = plt.contour(X, Y, P, levels=[config.PACT_KPA], linewidths=2)
#     c2 = plt.contour(X, Y, P, levels=[config.P_SAFE_MAX_KPA], linewidths=2)
#     plt.clabel(c1, fmt={config.PACT_KPA: f"Pact={config.PACT_KPA:.1f}"})
#     plt.clabel(c2, fmt={config.P_SAFE_MAX_KPA: f"Psafe={config.P_SAFE_MAX_KPA:.1f}"})
#     plt.xlabel("Vg (mL)")
#     plt.ylabel("Vsol (mL)")
#     plt.title(f"Safe operating window | {result['temperature_c']} °C")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(config.OUTPUT_FOLDER, f"safe_window_{int(result['temperature_c'])}C.png"),
#         dpi=300
#     )
#     plt.close()


# def plot_pressure_time_tradeoff(result):
#     X, Y = np.meshgrid(result["vg_vals"], result["vsol_vals"])
#     P = result["P_map"]
#     TACT = np.ma.masked_invalid(result["t_act_map"])

#     plt.figure(figsize=(8, 6))
#     contour = plt.contourf(X, Y, TACT, levels=25)
#     plt.colorbar(contour, label="Time to actuation (h)")
#     c = plt.contour(X, Y, P, levels=[5, 10, 20, 30, 40], linewidths=1)
#     plt.clabel(c, fmt="%.0f kPa")
#     plt.xlabel("Vg (mL)")
#     plt.ylabel("Vsol (mL)")
#     plt.title(f"Pressure-time tradeoff | {result['temperature_c']} °C")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(config.OUTPUT_FOLDER, f"pressure_time_tradeoff_{int(result['temperature_c'])}C.png"),
#         dpi=300
#     )
#     plt.close()


# def plot_regime_map(result):
#     X, Y = np.meshgrid(result["vg_vals"], result["vsol_vals"])
#     ratio = Y / X

#     regime = np.zeros_like(ratio, dtype=int)
#     regime[ratio < 1.0] = 0
#     regime[(ratio >= 1.0) & (ratio < 2.5)] = 1
#     regime[(ratio >= 2.5) & (ratio < 4.0)] = 2
#     regime[ratio >= 4.0] = 3

#     plt.figure(figsize=(8, 6))
#     plt.contourf(X, Y, regime, levels=[-0.5, 0.5, 1.5, 2.5, 3.5])
#     plt.xlabel("Vg (mL)")
#     plt.ylabel("Vsol (mL)")
#     plt.title(f"Regime map (ratio-based interpretation) | {result['temperature_c']} °C")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(config.OUTPUT_FOLDER, f"regime_map_{int(result['temperature_c'])}C.png"),
#         dpi=300
#     )
#     plt.close()


# def plot_design_recommendations(result):
#     vg_vals = result["vg_vals"]
#     vsol_vals = result["vsol_vals"]
#     P = result["P_map"]
#     TACT = result["t_act_map"]

#     rows = []
#     for i, vsol in enumerate(vsol_vals):
#         for j, vg in enumerate(vg_vals):
#             p = P[i, j]
#             t = TACT[i, j]
#             if np.isnan(t):
#                 continue
#             if p < config.PACT_KPA or p > config.P_SAFE_MAX_KPA:
#                 continue

#             score = t + 0.8 * vsol + 0.2 * vg
#             rows.append({
#                 "Vsol_ml": vsol,
#                 "Vg_ml": vg,
#                 "Pmax_kPa": p,
#                 "t_act_h": t,
#                 "score": score
#             })

#     if len(rows) == 0:
#         return

#     df = pd.DataFrame(rows).sort_values("score").head(config.N_TOP_RECOMMENDATIONS)
#     df.to_csv(
#         os.path.join(config.OUTPUT_FOLDER, f"recommendations_{int(result['temperature_c'])}C.csv"),
#         index=False
#     )

#     labels = [f"#{i+1}\nVsol={row.Vsol_ml:.1f}\nVg={row.Vg_ml:.1f}" for i, row in enumerate(df.itertuples())]

#     plt.figure(figsize=(9, 5))
#     plt.bar(labels, df["score"].values)
#     plt.ylabel("Recommendation score")
#     plt.title(f"Top design recommendations | {result['temperature_c']} °C")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(config.OUTPUT_FOLDER, f"design_recommendations_{int(result['temperature_c'])}C.png"),
#         dpi=300
#     )
#     plt.close()


# def plot_sensitivity_tornado(df_sum, sensor="ABP2"):
#     temp_c = config.SENSITIVITY_BASE_T_C
#     temp_label = config.MAP_TEMP_TO_LABEL[temp_c]
#     C0 = config.SENSITIVITY_BASE_C_MM
#     Vsol0 = config.SENSITIVITY_BASE_VSOL_ML
#     Vg0 = config.SENSITIVITY_BASE_VG_ML
#     frac = config.SENSITIVITY_DELTA_FRAC

#     beta_hat, k_hat, _ = estimate_beta_and_k(
#         df_sum=df_sum,
#         sensor=sensor,
#         temp_label=temp_label,
#         C_target_mM=C0
#     )

#     Tk0 = temp_c + 273.15
#     base_p = p_model_beta_max_kpa(Tk0, beta_hat, C0, Vsol0, Vg0)

#     params = {
#         "Temperature": (temp_c * (1 - frac), temp_c * (1 + frac)),
#         "Concentration": (C0 * (1 - frac), C0 * (1 + frac)),
#         "Vsol": (Vsol0 * (1 - frac), Vsol0 * (1 + frac)),
#         "Vg": (Vg0 * (1 - frac), Vg0 * (1 + frac)),
#         "Beta": (beta_hat * (1 - frac), beta_hat * (1 + frac)),
#         "k": (k_hat * (1 - frac), k_hat * (1 + frac)),
#     }

#     records = []

#     for name, (low, high) in params.items():
#         if name == "Temperature":
#             p_low = p_model_beta_max_kpa(low + 273.15, beta_hat, C0, Vsol0, Vg0)
#             p_high = p_model_beta_max_kpa(high + 273.15, beta_hat, C0, Vsol0, Vg0)
#         elif name == "Concentration":
#             p_low = p_model_beta_max_kpa(Tk0, beta_hat, low, Vsol0, Vg0)
#             p_high = p_model_beta_max_kpa(Tk0, beta_hat, high, Vsol0, Vg0)
#         elif name == "Vsol":
#             p_low = p_model_beta_max_kpa(Tk0, beta_hat, C0, low, Vg0)
#             p_high = p_model_beta_max_kpa(Tk0, beta_hat, C0, high, Vg0)
#         elif name == "Vg":
#             p_low = p_model_beta_max_kpa(Tk0, beta_hat, C0, Vsol0, low)
#             p_high = p_model_beta_max_kpa(Tk0, beta_hat, C0, Vsol0, high)
#         elif name == "Beta":
#             p_low = p_model_beta_max_kpa(Tk0, low, C0, Vsol0, Vg0)
#             p_high = p_model_beta_max_kpa(Tk0, high, C0, Vsol0, Vg0)
#         elif name == "k":
#             # k does not affect Pmax, but include time-to-actuation sensitivity instead
#             t_low = time_to_threshold_h(Tk0, beta_hat, C0, Vsol0, Vg0, low, config.PACT_KPA)
#             t_high = time_to_threshold_h(Tk0, beta_hat, C0, Vsol0, Vg0, high, config.PACT_KPA)
#             records.append({
#                 "parameter": name + " (time)",
#                 "low_change": t_low,
#                 "high_change": t_high,
#                 "base": time_to_threshold_h(Tk0, beta_hat, C0, Vsol0, Vg0, k_hat, config.PACT_KPA)
#             })
#             continue

#         records.append({
#             "parameter": name,
#             "low_change": p_low,
#             "high_change": p_high,
#             "base": base_p
#         })

#     df = pd.DataFrame(records)

#     # plot magnitude around base
#     plt.figure(figsize=(9, 5))
#     y = np.arange(len(df))

#     low_width = df["base"] - df["low_change"]
#     high_width = df["high_change"] - df["base"]

#     plt.barh(y, low_width, left=df["low_change"])
#     plt.barh(y, high_width, left=df["base"])
#     plt.yticks(y, df["parameter"])
#     plt.xlabel("Output change around base case")
#     plt.title("Sensitivity tornado chart")
#     plt.tight_layout()
#     plt.savefig(os.path.join(config.OUTPUT_FOLDER, "sensitivity_tornado.png"), dpi=300)
#     plt.close()


# def save_advanced_plots(results, df_val, curve_payloads, df_sum):
#     plot_validation_parity(df_val)
#     plot_m6_panel(curve_payloads)
#     plot_sensitivity_tornado(df_sum, sensor=config.MAP_SENSOR)

#     for result in results:
#         plot_safe_operating_window(result)
#         plot_pressure_time_tradeoff(result)
#         plot_regime_map(result)
#         plot_design_recommendations(result)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from physics_model import p_model_beta_max_kpa, time_to_threshold_h
from parameter_estimation import estimate_beta_and_k


# =========================================================
# VALIDATION PLOTS
# =========================================================
def plot_validation_parity(df_val):
    if len(df_val) == 0:
        return

    # Peak pressure parity
    x = df_val["pmax_meas_kpa"].values
    y = df_val["pmax_pred_curve_kpa"].values

    mn = min(np.min(x), np.min(y))
    mx = max(np.max(x), np.max(y))

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.plot([mn, mx], [mn, mx], "--")
    for _, row in df_val.iterrows():
        plt.annotate(row["sample"], (row["pmax_meas_kpa"], row["pmax_pred_curve_kpa"]))
    plt.xlabel("Measured peak pressure (kPa)")
    plt.ylabel("Predicted peak pressure (kPa)")
    plt.title("Validation parity plot: peak pressure")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_FOLDER, "validation_parity_peak_pressure.png"), dpi=300)
    plt.close()

    # Threshold time parity
    df_t = df_val.dropna(subset=["t_act_meas_h", "t_act_pred_h"]).copy()
    if len(df_t) > 0:
        x = df_t["t_act_meas_h"].values
        y = df_t["t_act_pred_h"].values
        mn = min(np.min(x), np.min(y))
        mx = max(np.max(x), np.max(y))

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y)
        plt.plot([mn, mx], [mn, mx], "--")
        for _, row in df_t.iterrows():
            plt.annotate(row["sample"], (row["t_act_meas_h"], row["t_act_pred_h"]))
        plt.xlabel("Measured threshold time (h)")
        plt.ylabel("Predicted threshold time (h)")
        plt.title("Validation parity plot: threshold time")
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_FOLDER, "validation_parity_threshold_time.png"), dpi=300)
        plt.close()


def plot_m6_panel(curve_payloads):
    if len(curve_payloads) == 0:
        return

    n = len(curve_payloads)
    fig, axes = plt.subplots(n, 1, figsize=(9, 4 * n), squeeze=False)

    for ax, payload in zip(axes[:, 0], curve_payloads):
        ax.plot(payload["t_h"], payload["P_meas_kpa"], label="Measured")
        ax.plot(payload["t_h"], payload["P_pred_kpa"], "--", label="Predicted")
        ax.axhline(payload["threshold_kpa"], linestyle=":", label="Threshold")
        ax.set_title(f"{payload['sample']} | {payload['sensor']} | {payload['temp_label']}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Pressure rise (kPa)")
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(config.OUTPUT_FOLDER, "m6_validation_panel.png"), dpi=300)
    plt.close(fig)


# =========================================================
# HELPERS FOR FIXED VTOT RESULTS
# =========================================================
def _build_design_dataframe(result):
    """
    Converts 1D fixed-Vtot sweep arrays into a clean dataframe.
    """
    vsol_vals = np.asarray(result["vsol_vals"], dtype=float)
    vg_vals = np.asarray(result["vg_vals"], dtype=float)
    p_vals = np.asarray(result["P_map"], dtype=float)
    t_vals = np.asarray(result["t_act_map"], dtype=float)

    df = pd.DataFrame({
        "Vsol_ml": vsol_vals,
        "Vg_ml": vg_vals,
        "Pmax_kPa": p_vals,
        "t_act_h": t_vals
    })

    df["Vtot_ml"] = df["Vsol_ml"] + df["Vg_ml"]
    df["fill_fraction"] = df["Vsol_ml"] / df["Vtot_ml"]

    return df


def _feasible_designs(result):
    df = _build_design_dataframe(result)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Pmax_kPa", "t_act_h"]).copy()

    df = df[
        (df["Pmax_kPa"] >= config.PACT_KPA) &
        (df["Pmax_kPa"] <= config.P_SAFE_MAX_KPA)
    ].copy()

    return df


def _recommendation_reason(row, df):
    if row["t_act_h"] <= df["t_act_h"].min() + 1e-9:
        return "Fastest actuation"
    if row["Vsol_ml"] <= df["Vsol_ml"].min() + 1e-9:
        return "Lowest reagent use"
    if row["Vg_ml"] >= df["Vg_ml"].max() - 1e-9:
        return "Largest headspace margin"
    if row["Pmax_kPa"] <= (config.P_SAFE_MAX_KPA * 0.8):
        return "Good safety margin"
    return "Balanced trade-off"


def _select_diverse_top_designs(result):
    """
    Pick top designs, but do not allow repeated Vg values.
    Since fixed Vtot makes Vg and Vsol linked, this also ensures diversity.
    """
    df = _feasible_designs(result)
    if len(df) == 0:
        return df

    # score: lower is better
    df["score"] = (
        df["t_act_h"]
        + 0.8 * df["Vsol_ml"]
        + 0.2 * df["Vg_ml"]
    )

    df = df.sort_values("score").copy()

    # diversity rule: do not repeat the same Vg
    df["Vg_round"] = df["Vg_ml"].round(2)
    df = df.drop_duplicates(subset="Vg_round", keep="first").copy()

    df = df.head(config.N_TOP_RECOMMENDATIONS).copy()
    df["reason"] = [_recommendation_reason(row, df) for _, row in df.iterrows()]

    return df


# =========================================================
# FIXED-VTOT INTERPRETATION PLOTS
# =========================================================
def plot_safe_operating_window(result):
    df = _build_design_dataframe(result)

    zone = np.zeros(len(df), dtype=int)
    zone[df["Pmax_kPa"] < config.PACT_KPA] = 0
    zone[(df["Pmax_kPa"] >= config.PACT_KPA) & (df["Pmax_kPa"] <= config.P_SAFE_MAX_KPA)] = 1
    zone[df["Pmax_kPa"] > config.P_SAFE_MAX_KPA] = 2

    plt.figure(figsize=(8, 5))
    plt.step(df["fill_fraction"], zone, where="mid")
    plt.axvline(df.loc[df["Pmax_kPa"] >= config.PACT_KPA, "fill_fraction"].min(), linestyle="--")
    plt.xlabel("Fill fraction = Vsol / Vtot")
    plt.ylabel("Zone (0=low, 1=safe, 2=too high)")
    plt.title(f"Safe operating window | {result['temperature_c']} °C | Vtot={config.VTOT_ML:.1f} mL")
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"safe_window_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


def plot_pressure_time_tradeoff(result):
    df = _build_design_dataframe(result)

    plt.figure(figsize=(8, 5))
    valid = df.dropna(subset=["t_act_h"]).copy()
    sc = plt.scatter(valid["fill_fraction"], valid["t_act_h"], c=valid["Pmax_kPa"])
    plt.colorbar(sc, label="Pmax (kPa)")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fill fraction = Vsol / Vtot")
    plt.ylabel("Time to actuation (h)")
    plt.title(f"Pressure-time tradeoff | {result['temperature_c']} °C")
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"pressure_time_tradeoff_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


def plot_regime_map(result):
    df = _build_design_dataframe(result)

    ratio = df["Vsol_ml"] / df["Vg_ml"]

    regime = np.zeros(len(df), dtype=int)
    regime[ratio < 1.0] = 0
    regime[(ratio >= 1.0) & (ratio < 2.5)] = 1
    regime[(ratio >= 2.5) & (ratio < 4.0)] = 2
    regime[ratio >= 4.0] = 3

    plt.figure(figsize=(8, 5))
    plt.step(df["fill_fraction"], regime, where="mid")
    plt.xlabel("Fill fraction = Vsol / Vtot")
    plt.ylabel("Regime index")
    plt.title(f"Regime map (ratio-based) | {result['temperature_c']} °C")
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"regime_map_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


# =========================================================
# RECOMMENDATIONS + PARETO STYLE PLOTS
# =========================================================
def plot_design_recommendations(result):
    df = _select_diverse_top_designs(result)
    if len(df) == 0:
        return

    df.to_csv(
        os.path.join(config.OUTPUT_FOLDER, f"recommendations_{int(result['temperature_c'])}C.csv"),
        index=False
    )

    labels = [
        f"#{i+1}\nVsol={row.Vsol_ml:.1f}\nVg={row.Vg_ml:.1f}\n{row.reason}"
        for i, row in enumerate(df.itertuples())
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, df["score"].values)
    plt.ylabel("Recommendation score")
    plt.title(f"Top diverse design recommendations | {result['temperature_c']} °C")
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"design_recommendations_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


def plot_pareto_scatter(result):
    """
    Standard Pareto-style scatter:
    time vs reagent use, colored by pressure.
    """
    df = _feasible_designs(result)
    if len(df) == 0:
        return

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(df["Vsol_ml"], df["t_act_h"], c=df["Pmax_kPa"])
    plt.colorbar(sc, label="Pmax (kPa)")
    plt.xlabel("Vsol (mL)")
    plt.ylabel("Time to actuation (h)")
    plt.title(f"Pareto-style tradeoff scatter | {result['temperature_c']} °C")
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"pareto_tradeoff_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


def plot_pareto_with_highlights(result):
    """
    Standard research-style tradeoff plot with recommended points highlighted.
    """
    df_all = _feasible_designs(result)
    df_top = _select_diverse_top_designs(result)

    if len(df_all) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(df_all["Vsol_ml"], df_all["t_act_h"], alpha=0.35, label="Feasible designs")

    if len(df_top) > 0:
        plt.scatter(df_top["Vsol_ml"], df_top["t_act_h"], label="Top picks")
        for i, row in enumerate(df_top.itertuples(), start=1):
            plt.annotate(f"#{i}", (row.Vsol_ml, row.t_act_h))

    plt.xlabel("Vsol (mL)")
    plt.ylabel("Time to actuation (h)")
    plt.title(f"Tradeoff plot with highlighted picks | {result['temperature_c']} °C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"pareto_highlighted_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


def plot_pressure_vs_fill_fraction(result):
    df = _build_design_dataframe(result)

    plt.figure(figsize=(7, 5))
    plt.plot(df["fill_fraction"], df["Pmax_kPa"])
    plt.axhline(config.PACT_KPA, linestyle="--", label="Pact")
    plt.axhline(config.P_SAFE_MAX_KPA, linestyle=":", label="Psafe")
    plt.xlabel("Fill fraction = Vsol / Vtot")
    plt.ylabel("Pmax (kPa)")
    plt.title(f"Pressure vs fill fraction | {result['temperature_c']} °C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"pressure_vs_fill_fraction_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


def plot_parallel_coordinates_top_designs(result):
    """
    A simple parallel-coordinates style plot for the top recommended designs.
    This is a common multi-objective visualization style.
    """
    df = _select_diverse_top_designs(result)
    if len(df) == 0:
        return

    # normalize columns for plotting
    cols = ["Vsol_ml", "Vg_ml", "Pmax_kPa", "t_act_h"]
    df_plot = df[cols].copy()

    for c in cols:
        cmin = df_plot[c].min()
        cmax = df_plot[c].max()
        if cmax > cmin:
            df_plot[c] = (df_plot[c] - cmin) / (cmax - cmin)
        else:
            df_plot[c] = 0.5

    x = np.arange(len(cols))

    plt.figure(figsize=(9, 5))
    for i, row in df_plot.iterrows():
        plt.plot(x, row.values, marker="o", label=f"Design {i+1}")

    plt.xticks(x, ["Vsol", "Vg", "Pmax", "t_act"])
    plt.ylabel("Normalized value")
    plt.title(f"Parallel-coordinates view of top designs | {result['temperature_c']} °C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"parallel_top_designs_{int(result['temperature_c'])}C.png"),
        dpi=300
    )
    plt.close()


# =========================================================
# SENSITIVITY
# =========================================================
def plot_sensitivity_tornado(df_sum, sensor="ABP2"):
    temp_c = config.SENSITIVITY_BASE_T_C
    temp_label = config.MAP_TEMP_TO_LABEL[temp_c]
    C0 = config.SENSITIVITY_BASE_C_MM
    Vsol0 = config.SENSITIVITY_BASE_VSOL_ML
    Vg0 = config.SENSITIVITY_BASE_VG_ML
    frac = config.SENSITIVITY_DELTA_FRAC

    beta_hat, k_hat, _ = estimate_beta_and_k(
        df_sum=df_sum,
        sensor=sensor,
        temp_label=temp_label,
        C_target_mM=C0
    )

    Tk0 = temp_c + 273.15
    base_p = p_model_beta_max_kpa(Tk0, beta_hat, C0, Vsol0, Vg0)

    params = {
        "Temperature": (temp_c * (1 - frac), temp_c * (1 + frac)),
        "Concentration": (C0 * (1 - frac), C0 * (1 + frac)),
        "Vsol": (Vsol0 * (1 - frac), Vsol0 * (1 + frac)),
        "Vg": (Vg0 * (1 - frac), Vg0 * (1 + frac)),
        "Beta": (beta_hat * (1 - frac), beta_hat * (1 + frac)),
        "k": (k_hat * (1 - frac), k_hat * (1 + frac)),
    }

    records = []

    for name, (low, high) in params.items():
        if name == "Temperature":
            p_low = p_model_beta_max_kpa(low + 273.15, beta_hat, C0, Vsol0, Vg0)
            p_high = p_model_beta_max_kpa(high + 273.15, beta_hat, C0, Vsol0, Vg0)
        elif name == "Concentration":
            p_low = p_model_beta_max_kpa(Tk0, beta_hat, low, Vsol0, Vg0)
            p_high = p_model_beta_max_kpa(Tk0, beta_hat, high, Vsol0, Vg0)
        elif name == "Vsol":
            p_low = p_model_beta_max_kpa(Tk0, beta_hat, C0, low, Vg0)
            p_high = p_model_beta_max_kpa(Tk0, beta_hat, C0, high, Vg0)
        elif name == "Vg":
            p_low = p_model_beta_max_kpa(Tk0, beta_hat, C0, Vsol0, low)
            p_high = p_model_beta_max_kpa(Tk0, beta_hat, C0, Vsol0, high)
        elif name == "Beta":
            p_low = p_model_beta_max_kpa(Tk0, low, C0, Vsol0, Vg0)
            p_high = p_model_beta_max_kpa(Tk0, high, C0, Vsol0, Vg0)
        elif name == "k":
            t_low = time_to_threshold_h(Tk0, beta_hat, C0, Vsol0, Vg0, low, config.PACT_KPA)
            t_high = time_to_threshold_h(Tk0, beta_hat, C0, Vsol0, Vg0, high, config.PACT_KPA)
            records.append({
                "parameter": name + " (time)",
                "low_change": t_low,
                "high_change": t_high,
                "base": time_to_threshold_h(Tk0, beta_hat, C0, Vsol0, Vg0, k_hat, config.PACT_KPA)
            })
            continue

        records.append({
            "parameter": name,
            "low_change": p_low,
            "high_change": p_high,
            "base": base_p
        })

    df = pd.DataFrame(records)

    plt.figure(figsize=(9, 5))
    y = np.arange(len(df))

    low_width = df["base"] - df["low_change"]
    high_width = df["high_change"] - df["base"]

    plt.barh(y, low_width, left=df["low_change"])
    plt.barh(y, high_width, left=df["base"])
    plt.yticks(y, df["parameter"])
    plt.xlabel("Output change around base case")
    plt.title("Sensitivity tornado chart")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_FOLDER, "sensitivity_tornado.png"), dpi=300)
    plt.close()


# =========================================================
# MASTER SAVE
# =========================================================
def save_advanced_plots(results, df_val, curve_payloads, df_sum):
    plot_validation_parity(df_val)
    plot_m6_panel(curve_payloads)
    plot_sensitivity_tornado(df_sum, sensor=config.MAP_SENSOR)

    for result in results:
        plot_safe_operating_window(result)
        plot_pressure_time_tradeoff(result)
        plot_regime_map(result)
        plot_design_recommendations(result)
        plot_pareto_scatter(result)
        plot_pareto_with_highlights(result)
        plot_pressure_vs_fill_fraction(result)
        plot_parallel_coordinates_top_designs(result)