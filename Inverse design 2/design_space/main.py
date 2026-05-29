"""
main.py — design space analysis
================================
Workflow (matches thesis_workflow.png):

  1. Build summary table  (η, k per experiment from training data)
  2. Validate M6 curves   (physics-informed model vs measured)
  3. Parameter exploration sweep  (Vsol, Vg, T grid)
  4. Pressure design maps (P_max heatmap)
  5. Design constraint    (A = n_gen / n_aq_max overlay)
  6. Actuation feasibility (P_act ≤ P_max ≤ P_safe  AND  t_act ≤ t_req)
"""

import os
import config
from data_loader import build_summary_table
from validation import validate_m6_curves
from parameter_sweep import run_all_sweeps
from maps import save_all_maps


def main():
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    # ── Step 1: build summary table ──────────────────────────────────────────
    print("=" * 60)
    print("Step 1: building summary table")
    df_sum = build_summary_table()
    print(df_sum[["sample", "sensor", "temp_label", "C_mM", "k", "eta_peak"]].to_string(index=False))

    # ── Step 2: validate M6 curves ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: M6 validation")
    df_val, _ = validate_m6_curves(df_sum, sensor=config.MAP_SENSOR)
    print(df_val[["sample", "temp_label", "eta_hat", "k_hat_mean",
                  "rmse_kpa", "pmax_meas_kpa", "pmax_pred_kpa"]].to_string(index=False))

    # ── Steps 3–6: sweeps + maps ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Steps 3–6: parameter sweep, pressure maps, design constraint, feasibility")
    results = run_all_sweeps(df_sum)

    for r in results:
        T_c = r["temperature_c"]
        A   = r["A_grid"]
        fg  = r["fgas_grid"]
        import numpy as np
        print(f"\n  T = {T_c} °C  η = {r['eta_hat']:.4f}  k = {r['k_hat']:.5f} h⁻¹")
        print(f"    A range:    [{np.nanmin(A):.2f}, {np.nanmax(A):.2f}]  (all > 1 → gas phase exists everywhere)")
        print(f"    f_gas range: [{np.nanmin(fg)*100:.1f}%, {np.nanmax(fg)*100:.1f}%]")

    save_all_maps(results)
    print(f"\nAll outputs saved to: {config.OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()
