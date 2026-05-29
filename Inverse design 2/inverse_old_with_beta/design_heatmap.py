"""
Two visualisations for the inverse-design framework:

1. plot_pressure_heatmap  — 2-D grid of (Vsol, Vg), coloured by Pmax.
                            Grey cells are infeasible (below Pact or above Psafe).
                            Step size is set coarse (HEATMAP_STEP_ML) so each
                            cell is a large square that is easy to read.

2. plot_vg_ladder         — For 5 increasing Vg values, shows the *best* Vsol
                            and the resulting Pmax / t_act.  Each row is
                            compared to the previous so the tradeoff is clear.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config
from helpers import make_range
from physics_model import p_model_beta_max_kpa, time_to_threshold_h

# ── Step size used ONLY for the heatmap visualisation ──────────────────────
# Coarser → bigger squares, easier to read.
# Change to 0.25 or 0.1 for a finer grid (more squares, smaller).
HEATMAP_STEP_ML = 0.5

# Vg values used for the ladder (5 spread-out values)
VG_LADDER = [0.5, 1.0, 2.0, 3.5, 5.0]   # mL


# ───────────────────────────────────────────────────────────────────────────
# 1.  PRESSURE HEATMAP
# ───────────────────────────────────────────────────────────────────────────

def plot_pressure_heatmap(result, temperature):
    """
    Heatmap of Pmax over the (Vsol, Vg) design space.
    Only cells where  Pact ≤ Pmax ≤ Psafe  are coloured.
    All other cells are drawn grey (infeasible).
    """
    Tk       = result["temperature_c"] + 273.15
    beta_hat = result["beta_hat"]
    k_hat    = result["k_hat"]
    C        = result["C_target_mM"]

    vsol_vals = make_range(0.5, 8.0, HEATMAP_STEP_ML)
    vg_vals   = make_range(0.5, 8.0, HEATMAP_STEP_ML)

    # Build (Vg × Vsol) pressure matrix — NaN where infeasible
    P_grid = np.full((len(vg_vals), len(vsol_vals)), np.nan)

    for i, vg in enumerate(vg_vals):
        for j, vsol in enumerate(vsol_vals):
            pmax = p_model_beta_max_kpa(Tk, beta_hat, C, vsol, vg)
            if config.PACT_KPA <= pmax <= config.P_SAFE_MAX_KPA:
                P_grid[i, j] = pmax

    # ── figure ──
    n_vsol = len(vsol_vals)
    n_vg   = len(vg_vals)
    cell   = 0.7                          # inches per cell
    fig, ax = plt.subplots(figsize=(n_vsol * cell + 2, n_vg * cell + 1.5))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#d0d0d0")         # grey for infeasible

    masked = np.ma.masked_invalid(P_grid)

    im = ax.pcolormesh(
        vsol_vals, vg_vals, masked,
        cmap=cmap,
        vmin=config.PACT_KPA,
        vmax=config.P_SAFE_MAX_KPA,
        shading="nearest"
    )

    # annotate each cell with its Pmax value
    for i, vg in enumerate(vg_vals):
        for j, vsol in enumerate(vsol_vals):
            val = P_grid[i, j]
            if not np.isnan(val):
                ax.text(vsol, vg, f"{val:.0f}",
                        ha="center", va="center",
                        fontsize=6.5, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("$P_{max}$ (kPa)", fontsize=10)

    # mark the constraint boundaries with lines
    ax.axhline(config.VG_MIN_ML, color="steelblue", ls="--", lw=1,
               label=f"$V_g$ min = {config.VG_MIN_ML} mL")

    infeasible_patch = mpatches.Patch(color="#d0d0d0", label="Infeasible")
    ax.legend(handles=[infeasible_patch], loc="upper right", fontsize=8)

    ax.set_xlabel("$V_{sol}$ (mL)", fontsize=11)
    ax.set_ylabel("$V_g$ (mL)",    fontsize=11)
    ax.set_title(
        f"$P_{{max}}$ heatmap  |  {temperature} °C  |  C = {C} mM\n"
        f"(step = {HEATMAP_STEP_ML} mL  ·  grey = infeasible)",
        fontsize=11
    )

    # force square cells
    ax.set_aspect("equal")
    ax.set_xticks(vsol_vals)
    ax.set_yticks(vg_vals)

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, f"heatmap_{temperature}C.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved heatmap_{temperature}C.png")


# ───────────────────────────────────────────────────────────────────────────
# 2.  VG LADDER
# ───────────────────────────────────────────────────────────────────────────

def _best_vsol_for_vg(vg, Tk, beta_hat, k_hat, C):
    """
    For a fixed Vg, sweep Vsol and return the design with the
    smallest t_act that still satisfies both pressure constraints.
    Returns None if no feasible Vsol exists.
    """
    vsol_vals = make_range(config.VSOL_MIN_ML, config.VSOL_MAX_ML, 0.1)
    best = None
    for vsol in vsol_vals:
        pmax = p_model_beta_max_kpa(Tk, beta_hat, C, vsol, vg)
        if not (config.PACT_KPA <= pmax <= config.P_SAFE_MAX_KPA):
            continue
        tact = time_to_threshold_h(Tk, beta_hat, C, vsol, vg, k_hat, config.PACT_KPA)
        if np.isnan(tact):
            continue
        if best is None or tact < best["t_act"]:
            best = {"Vg": vg, "Vsol": vsol, "Pmax": pmax, "t_act": tact}
    return best


def plot_vg_ladder(result, temperature):
    """
    Compare 5 Vg values (VG_LADDER) side-by-side.
    For each Vg the best Vsol (min t_act) is found.
    A delta row shows how each metric changes relative to the previous step.

    Also prints advice on step-size choice based on gradient magnitude.
    """
    Tk       = result["temperature_c"] + 273.15
    beta_hat = result["beta_hat"]
    k_hat    = result["k_hat"]
    C        = result["C_target_mM"]

    designs = []
    for vg in VG_LADDER:
        d = _best_vsol_for_vg(vg, Tk, beta_hat, k_hat, C)
        if d is not None:
            designs.append(d)

    if len(designs) < 2:
        print(f"  [vg_ladder] Not enough feasible Vg values at {temperature} °C.")
        return

    # ── console ladder ──────────────────────────────────────────────────────
    print(f"\n  === Vg ladder  |  {temperature} °C ===")
    print(f"  {'Vg':>5}  {'Vsol':>6}  {'Pmax':>8}  {'t_act':>8}  "
          f"{'ΔPmax':>8}  {'Δt_act':>9}  Step-size advice")
    print("  " + "-" * 75)

    for idx, d in enumerate(designs):
        if idx == 0:
            dpmax  = "—"
            dtact  = "—"
            advice = "  (reference)"
        else:
            prev   = designs[idx - 1]
            dvg    = d["Vg"] - prev["Vg"]
            dp     = d["Pmax"]  - prev["Pmax"]
            dt     = d["t_act"] - prev["t_act"]
            dpmax  = f"{dp:+.1f} kPa"
            dtact  = f"{dt:+.1f} h"

            # gradient-based step size advice
            grad_p = abs(dp) / dvg   # kPa per mL
            if grad_p > 5:
                advice = f"  ← steep ({grad_p:.1f} kPa/mL): use ≤0.1 mL steps"
            elif grad_p > 1:
                advice = f"  ← moderate ({grad_p:.1f} kPa/mL): 0.2–0.5 mL ok"
            else:
                advice = f"  ← gentle ({grad_p:.1f} kPa/mL): 0.5–1.0 mL ok"

        print(f"  {d['Vg']:>5.1f}  {d['Vsol']:>6.2f}  "
              f"{d['Pmax']:>7.1f}k  {d['t_act']:>7.1f}h  "
              f"{dpmax:>8}  {dtact:>9}  {advice}")

    # ── figure ──────────────────────────────────────────────────────────────
    vg_vals   = [d["Vg"]    for d in designs]
    pmax_vals = [d["Pmax"]  for d in designs]
    tact_vals = [d["t_act"] for d in designs]
    vsol_vals = [d["Vsol"]  for d in designs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(designs)))

    for ax, yvals, ylabel, fmt in zip(
        axes,
        [pmax_vals, tact_vals, vsol_vals],
        ["$P_{max}$ (kPa)", "Time to actuation (h)", "Optimal $V_{sol}$ (mL)"],
        [".1f", ".1f", ".2f"]
    ):
        bars = ax.bar(
            [str(v) for v in vg_vals], yvals,
            color=colors, edgecolor="black", linewidth=0.6, width=0.55
        )
        for bar, val in zip(bars, yvals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(yvals) * 0.02,
                format(val, fmt),
                ha="center", va="bottom", fontsize=8
            )
        ax.set_xlabel("$V_g$ (mL)", fontsize=10)
        ax.set_ylabel(ylabel,       fontsize=10)
        ax.set_ylim(0, max(yvals) * 1.18)

    # add threshold lines on Pmax panel
    axes[0].axhline(config.PACT_KPA,     color="green",  ls="--", lw=1, label=f"$P_{{act}}$={config.PACT_KPA} kPa")
    axes[0].axhline(config.P_SAFE_MAX_KPA, color="red",  ls="--", lw=1, label=f"$P_{{safe}}$={config.P_SAFE_MAX_KPA} kPa")
    axes[0].legend(fontsize=7)

    fig.suptitle(
        f"Vg ladder  |  {temperature} °C  |  C = {C} mM  "
        f"(each bar = best $V_{{sol}}$ for that $V_g$)",
        fontsize=11
    )
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, f"vg_ladder_{temperature}C.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved vg_ladder_{temperature}C.png")
