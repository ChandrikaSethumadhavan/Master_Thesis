import matplotlib.pyplot as plt
import os
import config


def plot_pareto(df, df_pareto, temperature):

    plt.figure(figsize=(7,6))

    plt.scatter(
        df["t_act"],
        df["Vsol"],
        alpha=0.3,
        label="All feasible designs"
    )

    plt.scatter(
        df_pareto["t_act"],
        df_pareto["Vsol"],
        color="red",
        label="Pareto optimal"
    )

    plt.xlabel("Time to actuation (h)")
    plt.ylabel("Solution volume (mL)")

    plt.title(f"Pareto front | {temperature} °C")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            config.OUTPUT_FOLDER,
            f"pareto_{temperature}C.png"
        ),
        dpi=300
    )

    plt.close()