"""
Three proper 2D design maps for the inverse-design framework:

  Map 1 — Pressure design map      (Vsol × Vg  →  Pmax, kPa)
  Map 2 — Actuation feasibility map (Vsol × Vg  →  zone: too-low / safe / too-high)
  Map 3 — Actuation time map        (Temperature × fill-fraction  →  t_act, h)

Maps 1 & 2 use an independent (Vsol, Vg) 2D sweep so the full geometry space is
visible, not just the Vtot = const diagonal.

Map 3 combines all three temperatures from `results` into one heatmap to show
how temperature interacts with geometry to determine response speed.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import config
from helpers import make_range
from physics_model import p_model_beta_max_kpa, time_to_threshold_h

# Grid resolution for maps 1 & 2
MAP2D_STEP_ML = 0.5   # mL  — increase to 1.0 for coarser/faster, 0.25 for finer


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: build full independent 2D (Vsol × Vg) pressure grid
# ─────────────────────────────────────────────────────────────────────────────
def _build_2d_grid(result, step=MAP2D_STEP_ML):
    Tk       = result["temperature_c"] + 273.15
    beta_hat = result["beta_hat"]
    k_hat    = result["k_hat"]
    C        = result["C_target_mM"]

    vsol_vals = make_range(0.5, 8.0, step)
    vg_vals   = make_range(0.5, 8.0, step)

    P_grid    = np.full((len(vg_vals), len(vsol_vals)), np.nan)
    T_grid    = np.full((len(vg_vals), len(vsol_vals)), np.nan)

    for i, vg in enumerate(vg_vals):
        for j, vsol in enumerate(vsol_vals):
            pmax = p_model_beta_max_kpa(Tk, beta_hat, C, vsol, vg)
            P_grid[i, j] = pmax

            if pmax >= config.PACT_KPA:
                tact = time_to_threshold_h(
                    Tk, beta_hat, C, vsol, vg, k_hat, config.PACT_KPA
                )
                T_grid[i, j] = tact

    return vsol_vals, vg_vals, P_grid, T_grid


# ─────────────────────────────────────────────────────────────────────────────
# MAP 1 — PRESSURE DESIGN MAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_pressure_design_map(result):
    """
    Clean continuous colour map of Pmax over the full (Vsol, Vg) geometry space.
    Takeaway: "what pressure does this geometry produce?"
    No overlaid lines — the colourbar and cell annotations tell the full story.
    """
    T = result["temperature_c"]
    C = result["C_target_mM"]

    vsol_vals, vg_vals, P_grid, _ = _build_2d_grid(result)
    Vsol_m, Vg_m = np.meshgrid(vsol_vals, vg_vals)

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.pcolormesh(
        Vsol_m, Vg_m, P_grid,
        cmap="plasma",
        vmin=0,
        vmax=config.P_SAFE_MAX_KPA * 1.1,
        shading="nearest"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$P_{max}$ (kPa)", fontsize=10)
    # mark Pact and Psafe on the colourbar only — no lines on the map
    cbar.ax.axhline(config.PACT_KPA,       color="white", lw=1.5, ls="--")
    cbar.ax.axhline(config.P_SAFE_MAX_KPA, color="white", lw=1.5, ls=":")
    cbar.ax.text(2.6, config.PACT_KPA,        f" $P_{{act}}$",  va="center", fontsize=7, color="white")
    cbar.ax.text(2.6, config.P_SAFE_MAX_KPA,  f" $P_{{safe}}$", va="center", fontsize=7, color="white")

    # annotate each cell with its Pmax value
    for i, vg in enumerate(vg_vals):
        for j, vsol in enumerate(vsol_vals):
            val = P_grid[i, j]
            if not np.isnan(val):
                ax.text(vsol, vg, f"{val:.0f}",
                        ha="center", va="center", fontsize=6, color="white")

    ax.set_xlabel("$V_{sol}$ (mL)", fontsize=11)
    ax.set_ylabel("$V_g$ (mL)",    fontsize=11)
    ax.set_title(
        f"Pressure design map  |  {T} °C  |  C = {C} mM\n"
        f"Geometry → maximum pressure produced (kPa)",
        fontsize=10
    )
    ax.set_aspect("equal")
    ax.set_xticks(vsol_vals)
    ax.set_yticks(vg_vals)

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, f"map1_pressure_{T}C.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved map1_pressure_{T}C.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAP 2 — ACTUATION FEASIBILITY MAP  (Geometry vs actuator threshold)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feasibility_map(result):
    """
    Flat three-zone classification: does this geometry meet the actuator threshold?
    Takeaway: "which geometries open the actuator, and which are dangerous or too weak?"

    This is deliberately categorical (no gradients, no lines) so it looks and reads
    completely differently from Map 1 (which is a continuous pressure surface).

      Zone 0 — Won't actuate   (Pmax < Pact)    → grey/blue, annotated "✗"
      Zone 1 — Safe to use     (Pact ≤ Pmax ≤ Psafe) → green,  annotated "✓"
      Zone 2 — Overpressure    (Pmax > Psafe)   → red/orange, annotated "!"
    """
    T = result["temperature_c"]
    C = result["C_target_mM"]

    vsol_vals, vg_vals, P_grid, _ = _build_2d_grid(result)
    Vsol_m, Vg_m = np.meshgrid(vsol_vals, vg_vals)

    zone = np.zeros_like(P_grid, dtype=int)
    zone[P_grid >= config.PACT_KPA]      = 1
    zone[P_grid > config.P_SAFE_MAX_KPA] = 2

    zone_colours = ["#9ecae1", "#41ab5d", "#f16913"]
    zone_symbols = {0: "✗", 1: "✓", 2: "!"}
    zone_labels  = [
        f"Won't actuate  ($P_{{max}} < {config.PACT_KPA:.0f}$ kPa)",
        f"Safe  ({config.PACT_KPA:.0f}–{config.P_SAFE_MAX_KPA:.0f} kPa)",
        f"Overpressure  ($P_{{max}} > {config.P_SAFE_MAX_KPA:.0f}$ kPa)",
    ]
    cmap_z = mcolors.ListedColormap(zone_colours)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.pcolormesh(
        Vsol_m, Vg_m, zone,
        cmap=cmap_z, vmin=-0.5, vmax=2.5,
        shading="nearest"
    )

    # annotate each cell with a symbol — NO contour lines, NO diagonal
    for i, vg in enumerate(vg_vals):
        for j, vsol in enumerate(vsol_vals):
            z = zone[i, j]
            ax.text(vsol, vg, zone_symbols[z],
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if z != 0 else "#1a6496")

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(zone_colours, zone_labels)]
    ax.legend(handles=patches, fontsize=8, loc="upper right",
              framealpha=0.9, edgecolor="grey")

    ax.set_xlabel("$V_{sol}$ (mL)", fontsize=11)
    ax.set_ylabel("$V_g$ (mL)",    fontsize=11)
    ax.set_title(
        f"Actuation feasibility map  |  {T} °C  |  C = {C} mM\n"
        f"Geometry → does it meet the actuator threshold?  "
        f"(✓ = safe, ✗ = too weak, ! = overpressure)",
        fontsize=9
    )
    ax.set_aspect("equal")
    ax.set_xticks(vsol_vals)
    ax.set_yticks(vg_vals)

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, f"map2_feasibility_{T}C.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved map2_feasibility_{T}C.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAP 3 — ACTUATION TIME MAP  (one panel per temperature)
# ─────────────────────────────────────────────────────────────────────────────

# Maps each temperature to its known ANT-EPO biological half-life (hours).
# None = no experimental half-life data available for that temperature.
_HALF_LIFE_H = {
    25.0: config.HALF_LIFE_RT_H,
    37.0: config.HALF_LIFE_37C_H,
    50.0: None,
}


def _min_tact_from_halflife(half_life_h):
    """
    Physical lower bound on t_act given the biological half-life.

    Derivation: the fastest possible actuation happens when Pmax is at its
    maximum allowed value (P_SAFE_MAX_KPA).  The fraction of reaction needed
    to reach Pact from that Pmax is:
        frac = 1 - Pact / Psafe
    With first-order kinetics  P(t) = Pmax * (1 - exp(-k*t)):
        t_act_min = -ln(frac) / k_bio   where k_bio = ln(2) / t_half
    """
    k_bio = np.log(2) / half_life_h
    frac  = 1.0 - config.PACT_KPA / config.P_SAFE_MAX_KPA   # e.g. 1 - 10/40 = 0.75
    return -np.log(frac) / k_bio


def plot_actuation_time_map(results):
    """
    Three side-by-side panels, one per temperature.

    Each panel:
      X-axis : fill fraction = Vsol / Vtot
      Y-axis : time to actuation (h)
      Line   : feasible designs coloured by Pmax (kPa)
      Grey band : fill fractions where Pmax < Pact (device won't open)
      Red dashed : physical minimum t_act derived from known half-life
                   — any value below this line violates the biology

    Reading the graph:
      • Moving right = more reagent loaded → faster actuation (line goes down)
      • Higher temperature panel → whole line shifts down (kinetics speed up)
      • The red dashed line is the speed limit set by the chemistry itself
      • Grey shading shows geometries that simply can't open the actuator
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    C = results[0]["C_target_mM"]

    for ax, result in zip(axes, results):
        T       = result["temperature_c"]
        vsol_arr = np.asarray(result["vsol_vals"], dtype=float)
        vg_arr   = np.asarray(result["vg_vals"],   dtype=float)
        p_arr    = np.asarray(result["P_map"],      dtype=float)
        t_arr    = np.asarray(result["t_act_map"],  dtype=float)

        ff_arr   = vsol_arr / (vsol_arr + vg_arr)   # fill fraction

        feasible = (
            (p_arr >= config.PACT_KPA) &
            (p_arr <= config.P_SAFE_MAX_KPA) &
            (~np.isnan(t_arr))
        )

        # ── grey shading for infeasible fill fractions ──
        if np.any(~feasible):
            for i, ff in enumerate(ff_arr):
                if not feasible[i]:
                    ax.axvspan(ff - 0.005, ff + 0.005,
                               color="#e0e0e0", alpha=0.6, lw=0)

        # ── feasible line coloured by Pmax ──
        if np.any(feasible):
            ff_f  = ff_arr[feasible]
            t_f   = t_arr[feasible]
            p_f   = p_arr[feasible]

            # sort by fill fraction for a clean line
            order = np.argsort(ff_f)
            ff_f, t_f, p_f = ff_f[order], t_f[order], p_f[order]

            # draw line first, then colour-coded scatter on top
            ax.plot(ff_f, t_f, color="steelblue", lw=1.2, alpha=0.5, zorder=2)
            sc = ax.scatter(ff_f, t_f, c=p_f, cmap="RdYlGn",
                            vmin=config.PACT_KPA, vmax=config.P_SAFE_MAX_KPA,
                            s=25, zorder=3)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
            cbar.set_label("$P_{max}$ (kPa)", fontsize=8)

        # ── physical minimum t_act from half-life ──
        hl = _HALF_LIFE_H.get(T)
        if hl is not None:
            t_min = _min_tact_from_halflife(hl)
            ax.axhline(t_min, color="red", ls="--", lw=1.8,
                       label=f"Min $t_{{act}}$ from $t_{{½}}$={hl} h\n"
                             f"= {t_min:.1f} h  (physics limit)")
            ax.legend(fontsize=7, loc="upper right")
            # shade the impossible region below the line
            ax.axhspan(0, t_min, color="red", alpha=0.06)

        # ── "too slow" reference ──
        ax.axhline(config.TACT_SLOW_H, color="grey", ls=":",
                   lw=1.2, label=f"Slow threshold ({config.TACT_SLOW_H} h)")

        ax.set_xlabel("Fill fraction  $V_{sol}/V_{tot}$", fontsize=10)
        ax.set_ylabel("Time to actuation (h)",            fontsize=10)
        ax.set_title(f"T = {T:.0f} °C", fontsize=13, fontweight="bold", pad=8)

        if np.any(feasible):
            y_ceil = min(np.nanmax(t_arr[feasible]) * 1.15, config.TACT_SLOW_H * 1.5)
            ax.set_ylim(bottom=0, top=y_ceil)

    fig.suptitle(
        f"Actuation time map  |  $V_{{tot}}$ = {config.VTOT_ML:.0f} mL  |  "
        f"C = {C} mM\n"
        "Line colour = $P_{{max}}$ (green=safe, red=overpressure)  |  "
        "Grey = fill fraction won't open actuator  |  "
        "Red dashed = physical speed limit from half-life",
        fontsize=9
    )

    plt.tight_layout()
    out = os.path.join(config.OUTPUT_FOLDER, "map3_actuation_time_vs_T.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("  Saved map3_actuation_time_vs_T.png")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CALL
# ─────────────────────────────────────────────────────────────────────────────
def save_design_maps(results):
    for result in results:
        plot_pressure_design_map(result)
        plot_feasibility_map(result)
    plot_actuation_time_map(results)
