"""
Temperature-aware inverse design entry point.

Run from inside inverse_temp/:
    python main_multiobjective.py

Differences from ../inverse/main_multiobjective.py:
  - Temperature T is a design variable (not fixed per run).
  - Arrhenius interpolation predicts k(T, C) for any T in [T_MIN_C, T_MAX_C].
  - Beta is linearly interpolated between RT and 37 °C groups.
  - Pareto objectives: [t_act, T_celsius, Vsol]  (all minimise)
  - Three corner solutions: fastest / coolest / least reagent
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inverse'))

import config   # loads inverse_temp/config.py (overrides parent)
os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

from data_loader import build_summary_table
from pareto_optimizer import run_pareto_search
from pareto_plots import plot_pareto, plot_temperature_tradeoff


def main():
    print("Building summary table from experimental CSV files...")
    df_sum = build_summary_table()

    print(f"\nRunning temperature-aware Pareto search")
    print(f"  Sensor        : {config.MAP_SENSOR}")
    print(f"  Concentration : {config.MAP_CONCENTRATION_MM} mM (fixed)")
    print(f"  T range       : {config.T_MIN_C}–{config.T_MAX_C} °C "
          f"(step = {config.T_STEP_C} °C)")
    print(f"  Vg range      : {config.VG_MIN_ML}–{config.VG_MAX_ML} mL")
    print(f"  Vsol range    : {config.VSOL_MIN_ML}–{config.VSOL_MAX_ML} mL")

    df, df_pareto, corners = run_pareto_search(
        df_sum,
        sensor=config.MAP_SENSOR,
        C_mM=config.MAP_CONCENTRATION_MM
    )

    if df.empty:
        print("\nNo feasible designs found. Check pressure constraints.")
        return

    print(f"\n  Feasible designs : {len(df)}")
    print(f"  Pareto-optimal   : {len(df_pareto)}")

    print("\n=== Pareto corner solutions ===")
    labels = {
        "min_tact": "Fastest actuation  — hot device, high Vsol, small Vg",
        "min_T":    "Coolest design     — room temperature, geometry compensates",
        "min_Vsol": "Least reagent      — small fill, higher T or smaller Vg compensates",
    }
    for key, description in labels.items():
        if key not in corners:
            continue
        pt = corners[key]
        print(f"\n  [{description}]")
        print(f"    T       = {pt['T_C']:.1f} °C")
        print(f"    Vsol    = {pt['Vsol']:.2f} mL")
        print(f"    Vg      = {pt['Vg']:.2f} mL")
        print(f"    Pmax    = {pt['Pmax']:.1f} kPa")
        print(f"    t_act   = {pt['t_act']:.2f} h")

    plot_pareto(df, df_pareto, corners)
    plot_temperature_tradeoff(df, corners)

    print(f"\nOutputs saved to: {config.OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()
