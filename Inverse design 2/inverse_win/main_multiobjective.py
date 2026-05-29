



from design_map import build_all_design_maps
from data_loader import build_summary_table

# from design_map import build_all_design_maps
# from data_loader import build_summary_table

from pareto_optimizer import run_pareto_search
from pareto_plots import plot_pareto


def main():

    df_sum = build_summary_table()

    results = build_all_design_maps(df_sum)

    for result in results:

        df, df_pareto = run_pareto_search(result)

        plot_pareto(
            df,
            df_pareto,
            result["temperature_c"]
        )

        print("\nPareto designs:")
        print(df_pareto.head())


if __name__ == "__main__":
    main()