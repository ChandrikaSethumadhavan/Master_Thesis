import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import config
from helpers import make_range
from physics_model import p_model_eta_max_kpa, time_to_threshold_h

_STEP = 0.5
_HALF_LIFE = {25.0: config.HALF_LIFE_RT_H, 37.0: config.HALF_LIFE_37C_H, 50.0: None}


def _build_2d_grid(result):
    Tk, eta, k, C = result["temperature_c"] + 273.15, result["eta_hat"], result["k_hat"], result["C_target_mM"]
    vsol = make_range(0.5, 8.0, _STEP)
    vg   = make_range(0.5, 8.0, _STEP)
    P = np.full((len(vg), len(vsol)), np.nan)
    T = np.full((len(vg), len(vsol)), np.nan)
    for i, g in enumerate(vg):
        for j, s in enumerate(vsol):
            P[i, j] = p_model_eta_max_kpa(Tk, eta, C, s, g)
            if P[i, j] >= config.PACT_KPA:
                T[i, j] = time_to_threshold_h(Tk, eta, C, s, g, k, config.PACT_KPA)
    return vsol, vg, P, T


def plot_pressure_design_map(result):
    T_c, C = result["temperature_c"], result["C_target_mM"]
    vsol, vg, P, _ = _build_2d_grid(result)
    Vm, Gm = np.meshgrid(vsol, vg)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(Vm, Gm, P, cmap="plasma", vmin=0, vmax=config.P_SAFE_MAX_KPA * 1.1, shading="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$\\Delta P_{max}$ (kPa)", fontsize=10)
    cbar.ax.axhline(config.PACT_KPA,       color="white", lw=1.5, ls="--")
    cbar.ax.axhline(config.P_SAFE_MAX_KPA, color="white", lw=1.5, ls=":")
    cbar.ax.text(2.6, config.PACT_KPA,       " $P_{act}$",  va="center", fontsize=7, color="white")
    cbar.ax.text(2.6, config.P_SAFE_MAX_KPA, " $P_{safe}$", va="center", fontsize=7, color="white")
    for i, g in enumerate(vg):
        for j, s in enumerate(vsol):
            if not np.isnan(P[i, j]):
                ax.text(s, g, f"{P[i,j]:.0f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_xlabel("$V_{sol}$ (mL)", fontsize=11); ax.set_ylabel("$V_g$ (mL)", fontsize=11)
    ax.set_title(f"Pressure design map  |  {T_c} °C  |  C = {C} mM\nEq 17: geometry → max pressure (kPa)", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks(vsol); ax.set_yticks(vg)
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, f"map1_pressure_{T_c}C.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved map1_pressure_{T_c}C.png")


def plot_feasibility_map(result):
    T_c, C = result["temperature_c"], result["C_target_mM"]
    vsol, vg, P, _ = _build_2d_grid(result)
    Vm, Gm = np.meshgrid(vsol, vg)
    zone = np.zeros_like(P, dtype=int)
    zone[P >= config.PACT_KPA]       = 1
    zone[P >  config.P_SAFE_MAX_KPA] = 2
    colours = ["#9ecae1", "#41ab5d", "#f16913"]
    symbols = {0: "✗", 1: "✓", 2: "!"}
    labels  = [f"Won't actuate (< {config.PACT_KPA:.0f} kPa)",
               f"Safe ({config.PACT_KPA:.0f}–{config.P_SAFE_MAX_KPA:.0f} kPa)",
               f"Overpressure (> {config.P_SAFE_MAX_KPA:.0f} kPa)"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(Vm, Gm, zone, cmap=mcolors.ListedColormap(colours), vmin=-0.5, vmax=2.5, shading="nearest")
    for i, g in enumerate(vg):
        for j, s in enumerate(vsol):
            z = zone[i, j]
            ax.text(s, g, symbols[z], ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if z != 0 else "#1a6496")
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)],
              fontsize=8, loc="upper right", framealpha=0.9, edgecolor="grey")
    ax.set_xlabel("$V_{sol}$ (mL)", fontsize=11); ax.set_ylabel("$V_g$ (mL)", fontsize=11)
    ax.set_title(f"Actuation feasibility map  |  {T_c} °C  |  C = {C} mM\n✓ = safe  ✗ = too weak  ! = overpressure", fontsize=10)
    ax.set_aspect("equal"); ax.set_xticks(vsol); ax.set_yticks(vg)
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, f"map2_feasibility_{T_c}C.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved map2_feasibility_{T_c}C.png")


def plot_actuation_time_map(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
    C = results[0]["C_target_mM"]

    for ax, r in zip(axes, results):
        T_c  = r["temperature_c"]
        vsol = np.asarray(r["vsol_vals"], float)
        vg   = np.asarray(r["vg_vals"],   float)
        P    = np.asarray(r["P_map"],     float)
        t    = np.asarray(r["t_act_map"], float)
        ff   = vsol / (vsol + vg)

        ok = (P >= config.PACT_KPA) & (P <= config.P_SAFE_MAX_KPA) & (~np.isnan(t))
        for i, f in enumerate(ff):
            if not ok[i]: ax.axvspan(f - 0.005, f + 0.005, color="#e0e0e0", alpha=0.6, lw=0)

        if ok.any():
            idx = np.argsort(ff[ok])
            sc = ax.scatter(ff[ok][idx], t[ok][idx], c=P[ok][idx], cmap="RdYlGn",
                            vmin=config.PACT_KPA, vmax=config.P_SAFE_MAX_KPA, s=25, zorder=3)
            ax.plot(ff[ok][idx], t[ok][idx], color="steelblue", lw=1.2, alpha=0.5, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
            cbar.set_label("$\\Delta P_{max}$ (kPa)", fontsize=8)

        hl = _HALF_LIFE.get(T_c)
        if hl:
            t_min = -np.log(1.0 - config.PACT_KPA / config.P_SAFE_MAX_KPA) / (np.log(2) / hl)
            ax.axhline(t_min, color="red", ls="--", lw=1.8, label=f"Min $t_{{act}}$ = {t_min:.1f} h")
            ax.axhspan(0, t_min, color="red", alpha=0.06)
            ax.legend(fontsize=7, loc="upper right")

        ax.axhline(config.TACT_SLOW_H, color="grey", ls=":", lw=1.2)
        ax.set_xlabel("Fill fraction $V_{sol}/V_{tot}$", fontsize=10)
        ax.set_ylabel("Time to actuation (h)", fontsize=10)
        ax.set_title(f"T = {T_c:.0f} °C", fontsize=13, fontweight="bold", pad=8)
        if ok.any():
            ax.set_ylim(0, min(np.nanmax(t[ok]) * 1.15, config.TACT_SLOW_H * 1.5))

    fig.suptitle(f"Actuation time map  |  $V_{{tot}}$ = {config.VTOT_ML:.0f} mL  |  C = {C} mM\n"
                 "Colour = $\\Delta P_{{max}}$  |  Grey = won't open  |  Red dashed = speed limit", fontsize=9)
    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, "map3_actuation_time_vs_T.png")
    plt.savefig(out, dpi=150); plt.close()
    print("  Saved map3_actuation_time_vs_T.png")


def save_design_maps(results):
    for r in results:
        plot_pressure_design_map(r)
        plot_feasibility_map(r)
    plot_actuation_time_map(results)
