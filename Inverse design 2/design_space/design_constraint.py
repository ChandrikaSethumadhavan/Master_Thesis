"""
Design constraint: A = n_gen / n_aq_max
=======================================
n_gen    = total O2 generated  = C0 * Vsol  [mol]
n_aq_max = O2 dissolved at P_max via Henry's law = H(T) * P_max * Vsol  [mol]

Substituting P_max = eta * n_gen / (Vg/(R*T) + H*Vsol):

    A = (Vg/(R*T) + H*Vsol) / (eta * H * Vsol)
      = 1/eta  +  Vg / (eta * R * T * H * Vsol)

Interpretation
--------------
A > 1  →  gas-phase exists  ✓ valid  (workflow criterion)
A ≤ 1  →  dissolution dominated  ✗ invalid

For any Vg > 0 and eta < 1, A > 1/eta > 1, so A > 1 is always satisfied
in the physical parameter space.  The A value itself is informative:

    small A (close to 1/eta)  →  small headspace, most O2 dissolves
    large A                   →  large headspace, gas-phase dominant

The gas fraction — what fraction of generated O2 ends up in the headspace:

    f_gas = eta * (Vg/(R*T)) / (Vg/(R*T) + H*Vsol)

f_gas → 0 as Vg → 0  (dissolution dominated, no pressure build-up)
f_gas → eta as H → 0  (no dissolution, all in gas)
"""

import numpy as np
import config
from battino import calc_H_SI
from helpers import ml_to_m3

R = config.R_J


def compute_A(Vsol_ml: float, Vg_ml: float, Tk: float, eta: float) -> float:
    """Return A = n_gen / n_aq_max for a single (Vsol, Vg, T, eta) point."""
    Vg_m3   = ml_to_m3(Vg_ml)
    Vsol_m3 = ml_to_m3(Vsol_ml)
    H = float(calc_H_SI(Tk))
    return (Vg_m3 / (R * Tk) + H * Vsol_m3) / (eta * H * Vsol_m3)


def compute_gas_fraction(Vsol_ml: float, Vg_ml: float, Tk: float, eta: float) -> float:
    """Return f_gas = fraction of generated O2 that ends up in the gas phase."""
    Vg_m3   = ml_to_m3(Vg_ml)
    Vsol_m3 = ml_to_m3(Vsol_ml)
    H = float(calc_H_SI(Tk))
    gas_cap = Vg_m3 / (R * Tk)          # mol/Pa — gas-phase capacity
    aq_cap  = H * Vsol_m3               # mol/Pa — aqueous capacity
    return eta * gas_cap / (gas_cap + aq_cap)


def build_A_grid(vsol_vals, vg_vals, Tk: float, eta: float):
    """Compute A and f_gas on a 2D (Vsol, Vg) grid.

    Returns
    -------
    A_grid     : shape (len(vg), len(vsol))
    fgas_grid  : shape (len(vg), len(vsol))
    """
    A_grid    = np.full((len(vg_vals), len(vsol_vals)), np.nan)
    fgas_grid = np.full((len(vg_vals), len(vsol_vals)), np.nan)
    for i, vg in enumerate(vg_vals):
        for j, vs in enumerate(vsol_vals):
            A_grid[i, j]    = compute_A(vs, vg, Tk, eta)
            fgas_grid[i, j] = compute_gas_fraction(vs, vg, Tk, eta)
    return A_grid, fgas_grid
