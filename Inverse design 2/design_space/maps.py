"""
Design maps
===========
map1  — Pressure design map  (P_max heatmap + A contours)
map2  — Design constraint map (A heatmap with gas-fraction scale)
map3  — Feasibility map  (P, time, A constraints combined)
map4  — Actuation time within the feasible safe zone
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import config


# ── helpers ──────────────────────────────────────────────────────────────────

def _mesh(vsol, vg):
    return np.meshgrid(vsol, vg)


def _out(name):
    return os.path.join(config.OUTPUT_FOLDER, name)


def _feasibility_zones(P_grid, T_grid, A_grid):
    """Return integer zone array:
        0 = dissolution dominated   (A ≤ 1)   — physics invalid
        1 = won't actuate           (P_max < P_act)
        2 = feasible / safe         (P_act ≤ P_max ≤ P_safe, t_act ≤ t_req)
        3 = overpressure            (P_max > P_safe)
        4 = feasible but too slow   (P_act ≤ P_max ≤ P_safe, t_act > t_req)
    """
    zone = np.ones_like(P_grid, dtype=int)               # default: won't actuate
    actuates = P_grid >= config.PACT_KPA
    overP    = P_grid >  config.P_SAFE_MAX_KPA

    safe      = actuates & ~overP
    fast      = safe & (T_grid <= config.TACT_REQ_H) & ~np.isnan(T_grid)
    slow      = safe & (~fast)

    zone[fast]           = 2    # feasible
    zone[overP]          = 3    # overpressure
    zone[slow]           = 4    # feasible pressure but too slow
    zone[A_grid <= 1.0]  = 0    # dissolution dominated (overrides all)

    return zone


# ── map 1: pressure design map ───────────────────────────────────────────────

def plot_pressure_map(result):
    T_c  = result["temperature_c"]
    C    = result["C_target_mM"]
    vsol = result["vsol_vals"]
    vg   = result["vg_vals"]
    P    = result["P_grid"]
    A    = result["A_grid"]

    Vm, Gm = _mesh(vsol, vg)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.pcolormesh(Vm, Gm, P, cmap="plasma",
                       vmin=0, vmax=config.P_SAFE_MAX_KPA * 1.1, shading="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\Delta P_{\max}$ (kPa)", fontsize=11)
    cbar.ax.axhline(config.PACT_KPA,       color="white", lw=1.5, ls="--")
    cbar.ax.axhline(config.P_SAFE_MAX_KPA, color="white", lw=1.5, ls=":")
    cbar.ax.text(2.6, config.PACT_KPA,       " $P_{act}$",  va="center", fontsize=8, color="white")
    cbar.ax.text(2.6, config.P_SAFE_MAX_KPA, " $P_{safe}$", va="center", fontsize=8, color="white")

    # annotate P_max values in each cell
    for i in range(len(vg)):
        for j in range(len(vsol)):
            if not np.isnan(P[i, j]):
                ax.text(vsol[j], vg[i], f"{P[i,j]:.0f}",
                        ha="center", va="center", fontsize=5.5, color="white")

    # A contour lines (design constraint overlay)
    A_levels = [2, 3, 5, 10]
    cs = ax.contour(Vm, Gm, A, levels=A_levels, colors="cyan", linewidths=0.9, linestyles="--")
    ax.clabel(cs, fmt={lv: f"A={lv}" for lv in A_levels}, fontsize=7, inline=True)

    ax.set_xlabel(r"$V_{sol}$ (mL)", fontsize=12)
    ax.set_ylabel(r"$V_g$ (mL)", fontsize=12)
    ax.set_title(
        f"Pressure design map  |  {T_c} °C  |  C = {C} mM\n"
        r"$\Delta P_{\max}$ (kPa) — cyan dashed = A contours (design constraint)",
        fontsize=10
    )
    ax.set_xticks(vsol); ax.set_yticks(vg)
    plt.tight_layout()
    fname = f"map1_pressure_{T_c}C.png"
    plt.savefig(_out(fname), dpi=150); plt.close()
    print(f"  Saved {fname}")


# ── map 2: design constraint A ───────────────────────────────────────────────

def plot_design_constraint_map(result):
    """Heatmap of A = n_gen / n_aq_max across the (Vsol, Vg) space.

    A > 1 everywhere in the physical grid; the colour shows how gas-dominant
    the design is.  Small A (close to 1/η) = mostly dissolved, tiny gas fraction.
    Large A = gas-phase dominant.
    """
    T_c   = result["temperature_c"]
    C     = result["C_target_mM"]
    eta   = result["eta_hat"]
    vsol  = result["vsol_vals"]
    vg    = result["vg_vals"]
    A     = result["A_grid"]
    fgas  = result["fgas_grid"]

    Vm, Gm = _mesh(vsol, vg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- left: A heatmap ---
    ax = axes[0]
    im = ax.pcolormesh(Vm, Gm, A, cmap="YlOrRd", shading="nearest",
                       vmin=1.0, vmax=np.nanpercentile(A, 95))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("A = n_gen / n_aq_max", fontsize=10)
    # A = 1 boundary (dissolution-dominated limit)
    ax.contour(Vm, Gm, A, levels=[1.0], colors="blue", linewidths=2, linestyles="-")
    A_min = float(np.nanmin(A))
    ax.text(0.97, 0.03, f"A_min = {A_min:.2f}\n(at Vg = {vg[0]:.1f} mL, Vsol = {vsol[-1]:.1f} mL)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    ax.set_xlabel(r"$V_{sol}$ (mL)", fontsize=11)
    ax.set_ylabel(r"$V_g$ (mL)", fontsize=11)
    ax.set_title(f"Design constraint A  |  {T_c} °C\n"
                 r"A > 1: gas-phase exists ✓   A ≤ 1: dissolution-dominated ✗", fontsize=10)
    ax.set_xticks(vsol[::2]); ax.set_yticks(vg[::2])

    # --- right: gas fraction heatmap ---
    ax = axes[1]
    im2 = ax.pcolormesh(Vm, Gm, fgas * 100, cmap="RdYlGn", shading="nearest",
                        vmin=0, vmax=100)
    cbar2 = fig.colorbar(im2, ax=ax, pad=0.02)
    cbar2.set_label(r"$f_{gas}$ = gas-phase O₂ fraction (%)", fontsize=10)

    for i in range(len(vg)):
        for j in range(len(vsol)):
            if not np.isnan(fgas[i, j]):
                ax.text(vsol[j], vg[i], f"{fgas[i,j]*100:.0f}%",
                        ha="center", va="center", fontsize=5.5, color="black")

    ax.set_xlabel(r"$V_{sol}$ (mL)", fontsize=11)
    ax.set_ylabel(r"$V_g$ (mL)", fontsize=11)
    ax.set_title(
        f"Gas-phase fraction  |  {T_c} °C  |  η = {eta:.3f}\n"
        r"$f_{gas} \to 0$: all O₂ dissolves (small $V_g$)   "
        r"$f_{gas} \to \eta$: gas dominant (large $V_g$)",
        fontsize=10
    )
    ax.set_xticks(vsol[::2]); ax.set_yticks(vg[::2])

    fig.suptitle(f"Design constraint  |  C = {C} mM  |  {T_c} °C", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"map2_design_constraint_{T_c}C.png"
    plt.savefig(_out(fname), dpi=150); plt.close()
    print(f"  Saved {fname}")


# ── map 3: full feasibility ───────────────────────────────────────────────────

def plot_feasibility_map(result):
    """Five-zone feasibility map combining P, time, and A constraints."""
    T_c  = result["temperature_c"]
    C    = result["C_target_mM"]
    vsol = result["vsol_vals"]
    vg   = result["vg_vals"]
    P    = result["P_grid"]
    T    = result["T_grid"]
    A    = result["A_grid"]

    Vm, Gm = _mesh(vsol, vg)
    zone = _feasibility_zones(P, T, A)

    colours = ["#d9534f",   # 0: dissolution dominated — red
               "#9ecae1",   # 1: won't actuate — light blue
               "#41ab5d",   # 2: feasible — green
               "#f16913",   # 3: overpressure — orange
               "#fee391"]   # 4: too slow — yellow
    labels = [
        "Dissolution dominated (A ≤ 1) — physics invalid",
        f"Won't actuate  (P_max < {config.PACT_KPA:.0f} kPa)",
        f"Feasible  ({config.PACT_KPA:.0f}–{config.P_SAFE_MAX_KPA:.0f} kPa, t ≤ {config.TACT_REQ_H:.0f} h)",
        f"Overpressure  (P_max > {config.P_SAFE_MAX_KPA:.0f} kPa)",
        f"Too slow  (t_act > {config.TACT_REQ_H:.0f} h)",
    ]
    symbols = {0: "✗", 1: "✗", 2: "✓", 3: "!", 4: "~"}

    cmap = mcolors.ListedColormap(colours)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.pcolormesh(Vm, Gm, zone, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest")

    for i in range(len(vg)):
        for j in range(len(vsol)):
            z = zone[i, j]
            col = "white" if z in (1, 2, 3, 4) else "white"
            ax.text(vsol[j], vg[i], symbols[z],
                    ha="center", va="center", fontsize=8, fontweight="bold", color=col)

    # A contour overlay — where A is small (close to 1/eta) mark with dashed contour
    eta = result["eta_hat"]
    A_min_practical = 1.0 / eta * 1.2     # 20% above the theoretical minimum
    cs = ax.contour(Vm, Gm, A, levels=[A_min_practical], colors=["#8856a7"],
                    linewidths=1.5, linestyles="--")
    ax.clabel(cs, fmt={A_min_practical: f"A={A_min_practical:.1f}\n(low gas frac.)"}, fontsize=7)

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, labels)]
    patches.append(mpatches.Patch(facecolor="none", edgecolor="#8856a7",
                                  linestyle="--", label=f"A = {A_min_practical:.1f} contour (low gas fraction)"))
    ax.legend(handles=patches, fontsize=7.5, loc="upper right",
              framealpha=0.92, edgecolor="grey")

    ax.set_xlabel(r"$V_{sol}$ (mL)", fontsize=12)
    ax.set_ylabel(r"$V_g$ (mL)", fontsize=12)
    ax.set_title(
        f"Actuation feasibility map  |  {T_c} °C  |  C = {C} mM\n"
        f"Constraints: P_act={config.PACT_KPA} kPa, P_safe={config.P_SAFE_MAX_KPA} kPa, "
        f"t_req={config.TACT_REQ_H} h, A > 1",
        fontsize=9.5
    )
    ax.set_xticks(vsol); ax.set_yticks(vg)
    plt.tight_layout()
    fname = f"map3_feasibility_{T_c}C.png"
    plt.savefig(_out(fname), dpi=150); plt.close()
    print(f"  Saved {fname}")


# ── map 4: actuation time ─────────────────────────────────────────────────────

def plot_actuation_time_map(result):
    """Heatmap of t_act within the safe-pressure zone."""
    T_c  = result["temperature_c"]
    C    = result["C_target_mM"]
    vsol = result["vsol_vals"]
    vg   = result["vg_vals"]
    P    = result["P_grid"]
    T    = result["T_grid"]
    A    = result["A_grid"]

    Vm, Gm = _mesh(vsol, vg)

    # Mask: only show t_act where design is in the safe-pressure window and A > 1
    safe_mask = (P >= config.PACT_KPA) & (P <= config.P_SAFE_MAX_KPA) & (A > 1.0)
    T_plot = np.where(safe_mask, T, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.pcolormesh(Vm, Gm, T_plot, cmap="RdYlGn_r", shading="nearest",
                       vmin=0, vmax=config.TACT_REQ_H)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Time to actuation (h)", fontsize=11)
    cbar.ax.axhline(config.TACT_REQ_H, color="black", lw=1.5, ls="--")
    cbar.ax.text(2.6, config.TACT_REQ_H, f" t_req={config.TACT_REQ_H:.0f} h",
                 va="center", fontsize=8)

    for i in range(len(vg)):
        for j in range(len(vsol)):
            if not np.isnan(T_plot[i, j]):
                col = "white" if T_plot[i, j] > config.TACT_REQ_H * 0.6 else "black"
                ax.text(vsol[j], vg[i], f"{T_plot[i,j]:.1f}",
                        ha="center", va="center", fontsize=5.5, color=col)

    # hatch regions outside the safe window
    outside = ~safe_mask
    ax.pcolormesh(Vm, Gm, np.where(outside, 1.0, np.nan),
                  cmap=mcolors.ListedColormap(["#d0d0d0"]), shading="nearest", alpha=0.5)

    ax.set_xlabel(r"$V_{sol}$ (mL)", fontsize=12)
    ax.set_ylabel(r"$V_g$ (mL)", fontsize=12)
    ax.set_title(
        f"Actuation time map  |  {T_c} °C  |  C = {C} mM\n"
        f"Only shown for safe zone ({config.PACT_KPA}–{config.P_SAFE_MAX_KPA} kPa) with A > 1",
        fontsize=10
    )
    ax.set_xticks(vsol); ax.set_yticks(vg)
    plt.tight_layout()
    fname = f"map4_actuation_time_{T_c}C.png"
    plt.savefig(_out(fname), dpi=150); plt.close()
    print(f"  Saved {fname}")


# ── entry point ───────────────────────────────────────────────────────────────

def save_all_maps(results: list[dict]):
    for r in results:
        print(f"\n  T = {r['temperature_c']} °C  |  η = {r['eta_hat']:.4f}  |  k = {r['k_hat']:.5f} h⁻¹")
        plot_pressure_map(r)
        plot_design_constraint_map(r)
        plot_feasibility_map(r)
        plot_actuation_time_map(r)
