"""
ANT-EPO Parylene Bellows Electrochemical Actuator — Pressure Design Space
Streamlit version of application_gui.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap
import streamlit as st

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

R_J = 8.314

# ── Henry's constant (Battino 1983) ───────────────────────────────────────────
_B_T  = np.array([273.15,278.15,283.15,288.15,293.15,298.15,303.15,308.15,
                   313.15,318.15,323.15,328.15,333.15,338.15,343.15,348.15])
_B_dH = np.array([-17.60,-16.18,-15.16,-14.15,-13.13,-12.11,-11.10,-10.08,
                    -9.06, -8.05, -7.03, -6.01, -5.00, -3.98, -2.97, -1.95])
_kH0, _dH0, _T0 = 1.3e-3, -12.11, 298.15

def H_battino(Tk):
    Tk   = np.asarray(Tk, float)
    dH_J = ((_dH0 + np.interp(Tk, _B_T, _B_dH)) / 2.0) * 1e3
    return _kH0 * np.exp(-(dH_J / R_J) * (1/Tk - 1/_T0)) * 1e3 / 101325.0

_K_TABLES = {
    "RT": np.array([
        [1.20, (0.03466 + 0.03725) / 2],
        [1.60, 0.01456],
        [2.00, 0.00744],
        [2.50, 0.02184],
        [2.67, (0.00536 + 0.00800 + 0.00865 + 0.00690) / 4],
        [3.10, 0.00872],
    ]),
    "37°C": np.array([
        [1.20, (0.08485 + 0.04386) / 2],
        [1.60, (0.06851 + 0.06152) / 2],
        [2.00, (0.09626 + 0.03499) / 2],
        [2.67, (0.09460 + 0.06720 + 0.10591 + 0.07205) / 4],
        [3.10, (0.05823 + 0.08253) / 2],
    ]),
    "45°C": np.array([[2.00,0.15122],[2.67,0.16076],[3.10,0.16272]]),
    "50°C": np.array([[2.00,0.20000],[2.67,0.23383],[3.10,0.28395]]),
}
_ETA_MEAN = {"RT": 0.877, "37°C": 0.915, "45°C": 0.995, "50°C": 0.990}
_EA_RT    = 43_700.0
_TEMP_K   = {"RT": 298.15, "37°C": 310.15, "45°C": 318.15, "50°C": 323.15}
_TEMP_C   = {"RT": 25.0,   "37°C": 37.0,   "45°C": 45.0,   "50°C": 50.0}

N_GRID  = 30
VSOL_UL = np.logspace(np.log10(0.05), np.log10(50), N_GRID)
VG_UL   = np.logspace(np.log10(0.05), np.log10(50), N_GRID)
_VS2D, _VG2D = np.meshgrid(VSOL_UL, VG_UL)
_NICE_V = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

_ZONE_COLORS = ["#FF8C00", "#FFD580", "#2d8a4e", "#2166ac"]
_CMAP_F = ListedColormap(_ZONE_COLORS)
_NORM_F = BoundaryNorm([0, 1, 2, 3, 4], _CMAP_F.N)


def _get_k_mean(tlbl):
    return float(np.mean(_K_TABLES[tlbl][:, 1]))

def _get_k_eff(tlbl, T_K):
    k = _get_k_mean(tlbl)
    if tlbl == "RT":
        k *= np.exp(-_EA_RT / R_J * (1.0 / T_K - 1.0 / 298.15))
    return k

def _pmax_2d(C_mM, eta, T_K):
    H  = H_battino(T_K)
    Vs = VSOL_UL[None, :] * 1e-9
    Vg = VG_UL[:, None]   * 1e-9
    return eta * C_mM * Vs / (Vg / (R_J * T_K) + Vs * H) / 1e3

def _tact_2d(Pmax_kPa, k_h, Pact_kPa):
    feasible = Pmax_kPa > Pact_kPa
    ratio    = np.clip(Pact_kPa / np.where(feasible, Pmax_kPa, 1.0), 0, 1 - 1e-9)
    return np.where(feasible, -np.log(1 - ratio) / max(k_h, 1e-9), np.nan)

def _zones(Pmax, tact, Pact, Psafe, treq):
    z = np.zeros(Pmax.shape, dtype=int)
    in_range = (Pmax >= Pact) & (Pmax <= Psafe)
    too_slow = np.nan_to_num(tact, nan=np.inf) > treq
    z[in_range & too_slow]  = 1
    z[in_range & ~too_slow] = 2
    z[Pmax > Psafe]         = 3
    return z


def _map_axis(ax, xlabel, ylabel, title):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(_NICE_V)
    ax.set_xticklabels([f"{v:g}" for v in _NICE_V])
    ax.set_yticks(_NICE_V)
    ax.set_yticklabels([f"{v:g}" for v in _NICE_V])
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, pad=7)
    ax.set_xlim(VSOL_UL[0] * 0.88, VSOL_UL[-1] * 1.12)
    ax.set_ylim(VG_UL[0]   * 0.88, VG_UL[-1]   * 1.12)

def _grid_dots(ax):
    ax.scatter(_VS2D.flat, _VG2D.flat, c="white", s=3,
               alpha=0.25, zorder=3, linewidths=0)

def _ref_star(ax, vsol_r, vg_r, label, color="white"):
    ax.scatter([vsol_r], [vg_r], c=color, s=220, marker="*",
               zorder=6, linewidths=0.4, edgecolors="#333333")
    ax.annotate(label, xy=(vsol_r, vg_r),
                xytext=(vsol_r * 0.32, vg_r * 2.8),
                color=color, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                zorder=7)


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANT-EPO Pressure Design Space",
    layout="wide",
)

st.markdown(
    "<h2 style='text-align:center; margin-bottom:0;'>"
    "Parylene Bellows Electrochemical Actuator · Pressure Design Space"
    "</h2>"
    "<p style='text-align:center; font-style:italic; margin-top:4px;'>"
    "Intraocular Drug Delivery Application "
    "(ANT-EPO enzymatic O₂ generation · Li et al., PMC3035913)"
    "</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Temperature group")
    tlbl = st.radio("", ["RT", "37°C", "45°C", "50°C"],
                    horizontal=True, label_visibility="collapsed")

    st.markdown("### Inputs")
    C      = st.slider("ANT-EPO concentration  C  (mM)",      1.0,  5.0,  2.0, 0.05)
    Pact   = st.slider("Min. actuation pressure  P_act  (kPa)", 0.5, 10.0, 1.72, 0.05)
    Psafe  = st.slider("Max. safe pressure  P_safe  (kPa)",     1.0, 50.0,10.0, 0.5)
    treq   = st.slider("Time requirement  t_req  (h)",         10.0,500.0,100.0,10.0)
    Vsol_r = st.slider("Reference  V_sol  (µL)",                0.05,50.0, 40.0,0.05)
    Vg_r   = st.slider("Reference  V_g   (µL)",                 0.05,50.0,  6.0,0.05)

T_K = _TEMP_K[tlbl]
T_C = _TEMP_C[tlbl]
eta = _ETA_MEAN[tlbl]
H   = H_battino(T_K)
k_h = _get_k_eff(tlbl, T_K)

Pmax  = _pmax_2d(C, eta, T_K)
tact  = _tact_2d(Pmax, k_h, Pact)
zones = _zones(Pmax, tact, Pact, Psafe, treq)

ref_j    = int(np.argmin(np.abs(VSOL_UL - Vsol_r)))
ref_i    = int(np.argmin(np.abs(VG_UL   - Vg_r)))
Pmax_ref = float(Pmax[ref_i, ref_j])
tact_ref = float(tact[ref_i, ref_j])
zone_ref = int(zones[ref_i, ref_j])
zone_lbl = ["Underpressure", "Too slow", "Feasible", "Overpressure"][zone_ref]

# ── Layout: info column + plots column ───────────────────────────────────────
col_info, col_plots = st.columns([1, 4])

with col_info:
    tact_str = f"{tact_ref:.1f} h" if np.isfinite(tact_ref) else "(infeasible)"
    feas_pct = float(np.mean(zones == 2)) * 100
    st.markdown(
        f"""
**Application**
Parylene Bellows Electrochemical Actuator
Li et al. (2011), PMC3035913

Device: Parylene-C bellows pump
Target: Intraocular drug delivery (glaucoma)

**Literature targets**
- P_act = 1.72–3.44 kPa
- V_ch ≈ 46 µL (30 µm, 1.5 conv)
- Flow ≈ 6.5 µL/min @ 1 mA

**Current model**
- T = {T_C:.1f} °C  [{tlbl}]
- k_mean = {k_h:.5f} h⁻¹
- η = {eta:.3f}
- H(T) = {H*1e5:.3f}×10⁻⁵

**Reference point**
- V_sol = {Vsol_r:.2g} µL
- V_g = {Vg_r:.2g} µL
- ΔP_max = {Pmax_ref:.2f} kPa
- t_act = {tact_str}
- Zone = **{zone_lbl}**

Feasible grid: {feas_pct:.0f}% of cells
"""
    )

with col_plots:
    fig = plt.figure(figsize=(14, 9), facecolor="#f8f8f8")
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.30,
                           height_ratios=[1.55, 1])
    ax_pm = fig.add_subplot(gs[0, 0])
    ax_fm = fig.add_subplot(gs[0, 1])
    ax_pt = fig.add_subplot(gs[1, :])

    # Pressure design map
    pcm = ax_pm.pcolormesh(VSOL_UL, VG_UL, Pmax, cmap="plasma",
                           shading="auto", vmin=0, vmax=Pmax.max())
    fig.colorbar(pcm, ax=ax_pm, label="$\\Delta P_{max}$ (kPa)",
                 extend="max", fraction=0.046, pad=0.04)
    _grid_dots(ax_pm)
    _ref_star(ax_pm, VSOL_UL[ref_j], VG_UL[ref_i], f"{Pmax_ref:.1f} kPa")
    _map_axis(ax_pm, "$V_{sol}$ (µL)", "$V_g$ (µL)",
              f"Pressure Design Map  |  {T_C:.1f} °C  |  "
              f"C = {C:.2f} mM,  $\\eta$ = {eta:.3f}")

    # Feasibility map
    ax_fm.pcolormesh(VSOL_UL, VG_UL, zones,
                     cmap=_CMAP_F, norm=_NORM_F, shading="auto")
    _grid_dots(ax_fm)
    _ref_star(ax_fm, VSOL_UL[ref_j], VG_UL[ref_i], zone_lbl, color="white")
    patches = [
        mpatches.Patch(color="#FF8C00",
                       label=f"Underpressure  ($\\Delta P$ < {Pact:.1f} kPa)"),
        mpatches.Patch(color="#FFD580",
                       label=f"Too slow  ($t_{{act}}$ > {treq:.0f} h)"),
        mpatches.Patch(color="#2d8a4e",
                       label=f"Feasible  ({Pact:.1f}–{Psafe:.1f} kPa,"
                             f"  $t$ ≤ {treq:.0f} h)  ✓"),
        mpatches.Patch(color="#2166ac",
                       label=f"Overpressure  ($\\Delta P$ > {Psafe:.1f} kPa)"),
    ]
    ax_fm.legend(handles=patches, loc="upper left",
                 framealpha=0.88, edgecolor="#aaaaaa")
    _map_axis(ax_fm, "$V_{sol}$ (µL)", "$V_g$ (µL)",
              f"Actuation Feasibility  |  {T_C:.1f} °C  |  "
              f"$P_{{act}}$ = {Pact:.1f},  $P_{{safe}}$ = {Psafe:.1f} kPa,  "
              f"$t_{{req}}$ = {treq:.0f} h")

    # P(t) curve
    Vs_m3 = Vsol_r * 1e-9
    Vg_m3 = Vg_r   * 1e-9
    t_end = max(treq * 3.5, 50.0)
    t_arr = np.linspace(0, t_end, 1500)
    P_arr = eta * C * Vs_m3 * (1 - np.exp(-k_h * t_arr)) / \
            (Vg_m3 / (R_J * T_K) + Vs_m3 * H) / 1e3

    ax_pt.plot(t_arr, P_arr, color="#2166ac", lw=2.2,
               label="$\\Delta P(t)$", zorder=4)
    ax_pt.axhline(Pact,  color="#FF8C00", lw=1.6, ls="--",
                  label=f"$P_{{act}}$ = {Pact:.2f} kPa", zorder=3)
    ax_pt.axhline(Psafe, color="#cc3333", lw=1.4, ls=":",
                  label=f"$P_{{safe}}$ = {Psafe:.1f} kPa", zorder=3)
    ax_pt.axhspan(Pact, Psafe, alpha=0.08, color="#2d8a4e", zorder=2)
    ax_pt.axhline(P_arr[-1], color="#AA3377", lw=1.2, ls="--", alpha=0.6,
                  label=f"$\\Delta P_{{max}}$ = {P_arr[-1]:.2f} kPa", zorder=3)
    if np.isfinite(tact_ref) and 0 < tact_ref < t_end:
        ax_pt.axvline(tact_ref, color="#FF8C00", lw=1.3, ls=":", alpha=0.85,
                      label=f"$t_{{act}}$ = {tact_ref:.1f} h", zorder=3)
        ax_pt.scatter([tact_ref], [Pact], color="#FF8C00", s=60, zorder=5)
    ax_pt.set_xlabel("Time (h)", labelpad=5)
    ax_pt.set_ylabel("$\\Delta P$ (kPa)", labelpad=5)
    ax_pt.set_title(
        f"Pressure vs Time  |  {T_C:.1f} °C  |  "
        f"$V_{{sol}}$ = {Vsol_r:.2g} µL,  $V_g$ = {Vg_r:.2g} µL",
        pad=6)
    ax_pt.legend(loc="lower right", framealpha=0.85)
    ax_pt.grid(True, alpha=0.22)
    ax_pt.set_xlim(0, t_end)
    ax_pt.set_ylim(0, max(P_arr.max(), Psafe) * 1.18)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.markdown(
    """
---
**Limitations**

1. k = mean across training concentrations. Assumes k is concentration-independent.
2. k and η from mL-scale experiments; µL actuators may differ.
3. ΔP_max depends on Vsol/Vg ratio, not absolute volume.
4. Constant temperature assumed.
5. η held at group mean; sealed µL devices may have different effective η.
"""
)
