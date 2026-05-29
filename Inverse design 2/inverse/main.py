import os
import config
from data_loader import build_summary_table
from design_map import build_all_design_maps
from validation import validate_m6_curves
from design_maps_2d import save_design_maps


def main():
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    df_sum = build_summary_table()
    print(df_sum[["sample", "sensor", "temp_label", "C_mM", "k", "eta_peak"]].to_string(index=False))

    df_val, _ = validate_m6_curves(df_sum, sensor=config.MAP_SENSOR)
    print(df_val[["sample", "temp_label", "eta_hat", "k_hat_mean", "rmse_kpa",
                  "pmax_meas_kpa", "pmax_pred_kpa"]].to_string(index=False))

    results = build_all_design_maps(df_sum)
    for r in results:
        print(f"  {r['temperature_c']} °C  η={r['eta_hat']:.4f}  k={r['k_hat']:.5f} h⁻¹  "
              f"train={list(r['train_df']['sample'].values)}")

    save_design_maps(results)
    print(f"\nOutputs: {config.OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()
