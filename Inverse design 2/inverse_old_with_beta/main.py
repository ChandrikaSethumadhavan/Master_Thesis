import os
import config
from data_loader import build_summary_table
from design_map import build_all_design_maps
from plotting import save_all_plots
from validation import validate_m6_curves
from advanced_plots import save_advanced_plots


def main():
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    print("\nBuilding summary table from experimental CSV files...")
    df_sum = build_summary_table()

    print("\nSummary table:")
    print(df_sum)

    print("\nRunning held-out validation on M6...")
    df_val, curve_payloads = validate_m6_curves(df_sum, sensor=config.MAP_SENSOR)

    print("\nValidation summary:")
    print(df_val)

    print("\nBuilding design maps...")
    results = build_all_design_maps(df_sum)

    for result in results:
        print("\n" + "=" * 70)
        print(f"Map temperature               : {result['temperature_c']} °C")
        print(f"Training label used           : {result['temp_label_used_for_training']}")
        print(f"Sensor                        : {result['sensor']}")
        print(f"Fixed concentration           : {result['C_target_mM']} mM")
        print(f"Predicted beta_hat            : {result['beta_hat']:.6f}")
        print(f"Predicted k_hat from GPR      : {result['k_hat']:.6f} 1/h")
        print("Training rows used:")
        print(result["train_df"][["sample", "C_mM", "k", "beta_peak"]])

    print("\nSaving standard plots...")
    save_all_plots(results)

    print("\nSaving advanced plots...")
    save_advanced_plots(results, df_val, curve_payloads, df_sum)

    print(f"\nDone. Outputs saved in: {config.OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()