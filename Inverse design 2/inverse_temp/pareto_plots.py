import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inverse'))

import matplotlib.pyplot as plt
import numpy as np
import config   # inverse_temp/config.py


def plot_pareto(df, df_pareto, corners):
    """
    Scatter: t_act (x) vs T_celsius (y), coloured by Vg.
    Design variables are now (Vsol, Vg) independently — no Vtot column exists.
    """
    if df.empty:
        print("  [pareto_plots] No feasible designs — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # all feasible designs — colour encodes Vg
    sc = ax.scatter(
        df["t_act"], df["T_C"],
        c=df["Vg"], cmap="YlGnBu",
        alpha=0.20, s=12, label="All feasible"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$V_g$ (mL)")

    # Pareto front
    ax.scatter(
        df_pareto["t_act"], df_pareto["T_C"],
        c=df_pareto["Vg"], cmap="YlGnBu",
        edgecolors="black", linewidths=0.6,
        s=50, zorder=3, label="Pareto front"
    )

    # three named corners
    corner_styles = {
        "min_tact": ("*", "tab:red",   "Min $t_{act}$ — fastest (hot, high fill)"),
        "min_T":    ("s", "tab:blue",  "Min $T$ — coolest (room-temp design)"),
        "min_Vsol": ("v", "tab:green", "Min $V_{sol}$ — least reagent"),
    }
    for key, (marker, color, label) in corner_styles.items():
        if key not in corners:
            continue
        pt = corners[key]
        ax.scatter(
            pt["t_act"], pt["T_C"],
            marker=marker, color=color, s=220, zorder=5,
            label=(f"{label}\n"
                   f"  $V_{{sol}}$={pt['Vsol']:.2f} mL, "
                   f"$V_g$={pt['Vg']:.2f} mL, "
                   f"T={pt['T_C']:.0f} °C, "
                   f"$t_{{act}}$={pt['t_act']:.1f} h")
        )

    ax.set_xlabel("Time to actuation (h)", fontsize=11)
    ax.set_ylabel("Temperature (°C)",      fontsize=11)
    ax.set_title(
        f"Temperature-aware Pareto front\n"
        f"C = {config.MAP_CONCENTRATION_MM} mM  |  "
        f"T range: {config.T_MIN_C:.0f}–{config.T_MAX_C:.0f} °C  |  "
        f"$V_g$: {config.VG_MIN_ML:.1f}–{config.VG_MAX_ML:.1f} mL  |  "
        f"$V_{{sol}}$: {config.VSOL_MIN_ML:.1f}–{config.VSOL_MAX_ML:.1f} mL",
        fontsize=10
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, "pareto_temperature.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("  Saved pareto_temperature.png")


def plot_temperature_tradeoff(df, corners):
    """
    Three-panel summary showing how t_act, Pmax, and Vsol each vary with
    temperature across all feasible (Vsol, Vg) designs.
    Each dot = one (Vsol, Vg, T) combination that passed the pressure constraints.
    """
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    temps   = sorted(df["T_C"].unique())
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(temps)))

    metrics = ["t_act", "Pmax",          "Vsol"]
    ylabels = ["$t_{act}$ (h)", "$P_{max}$ (kPa)", "$V_{sol}$ (mL)"]

    corner_styles = {
        "min_tact": ("*", "tab:red"),
        "min_T":    ("s", "tab:blue"),
        "min_Vsol": ("v", "tab:green"),
    }

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for t_c, color in zip(temps, palette):
            sub = df[df["T_C"] == t_c]
            ax.scatter(sub["T_C"], sub[metric],
                       color=color, alpha=0.25, s=8)

        for key, (marker, color) in corner_styles.items():
            if key not in corners:
                continue
            pt = corners[key]
            ax.scatter(pt["T_C"], pt[metric],
                       marker=marker, color=color, s=160, zorder=5)

        ax.set_xlabel("Temperature (°C)", fontsize=10)
        ax.set_ylabel(ylabel,             fontsize=10)

    axes[0].axhline(config.TACT_SLOW_H,    color="grey",  ls="--", lw=1,
                    label=f"Slow threshold ({config.TACT_SLOW_H} h)")
    axes[0].legend(fontsize=7)
    axes[1].axhline(config.PACT_KPA,       color="green", ls="--", lw=1)
    axes[1].axhline(config.P_SAFE_MAX_KPA, color="red",   ls="--", lw=1)

    fig.suptitle(
        "Effect of temperature on design metrics  |  all feasible (Vsol, Vg) combos\n"
        "★ = fastest   ■ = coolest   ▼ = least reagent",
        fontsize=10
    )
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, "temperature_tradeoff.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("  Saved temperature_tradeoff.png")
