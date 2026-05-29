import os
import config
from design_map import build_all_design_maps
from data_loader import build_summary_table
from pareto_optimizer import run_pareto_search
from pareto_plots import plot_pareto
from design_heatmap import plot_pressure_heatmap, plot_vg_ladder
from design_maps_2d import save_design_maps


def main():
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    df_sum  = build_summary_table()
    results = build_all_design_maps(df_sum)

    for result in results:
        T = result["temperature_c"]
        print(f"\n{'='*60}")
        print(f"Temperature: {T} °C")

        df, df_pareto, corners = run_pareto_search(result)

        if df.empty:
            print("  No feasible designs found.")
            continue

        print(f"  Feasible designs : {len(df)}")
        print(f"  Pareto-optimal   : {len(df_pareto)}")

        print("\n  === Pareto corner solutions ===")
        labels = {
            "min_tact": "Fastest actuation  (min t_act) — high fill, small Vg",
            "min_Vtot": "Most compact       (min Vtot)  — smallest total chamber",
            "max_Vg":   "Pressure-safe      (max Vg)    — large headspace buffer",
        }
        for key, description in labels.items():
            if key not in corners:
                continue
            pt = corners[key]
            print(f"\n  [{description}]")
            print(f"    Vsol    = {pt['Vsol']:.2f} mL")
            print(f"    Vg      = {pt['Vg']:.2f} mL")
            print(f"    Vtot    = {pt['Vtot']:.2f} mL")
            print(f"    Pmax    = {pt['Pmax']:.1f} kPa")
            print(f"    t_act   = {pt['t_act']:.2f} h")

        plot_pareto(df, df_pareto, corners, T)
        plot_pressure_heatmap(result, T)
        plot_vg_ladder(result, T)

    print("\nGenerating 2D design maps...")
    save_design_maps(results)


if __name__ == "__main__":
    main()
