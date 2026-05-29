import matplotlib.pyplot as plt
import os
import config


def plot_pareto(df, df_pareto, corners, temperature):
    if df.empty:
        print(f"  [pareto_plots] No feasible designs at {temperature} °C — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # All feasible designs — colour encodes Vg
    sc = ax.scatter(
        df["t_act"], df["Vtot"],
        c=df["Vg"], cmap="viridis",
        alpha=0.20, s=12,
        label="All feasible"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$V_g$ (mL)")

    # Full Pareto front
    ax.scatter(
        df_pareto["t_act"], df_pareto["Vtot"],
        c=df_pareto["Vg"], cmap="viridis",
        edgecolors="black", linewidths=0.7,
        s=60, zorder=3,
        label="Pareto front"
    )

    # Three corner solutions
    corner_styles = {
        "min_tact": ("*", "tab:red",    "Min $t_{act}$ — fastest"),
        "min_Vtot": ("v", "tab:blue",   "Min $V_{tot}$ — compact"),
        "max_Vg":   ("^", "tab:green",  "Max $V_g$ — pressure-safe"),
    }

    for key, (marker, color, label) in corner_styles.items():
        if key not in corners:
            continue
        pt = corners[key]
        ax.scatter(
            pt["t_act"], pt["Vtot"],
            marker=marker, color=color, s=220,
            zorder=5,
            label=(
                f"{label}\n"
                f"  $V_{{sol}}$={pt['Vsol']:.2f} mL, "
                f"$V_g$={pt['Vg']:.2f} mL, "
                f"$V_{{tot}}$={pt['Vtot']:.1f} mL, "
                f"$t_{{act}}$={pt['t_act']:.1f} h"
            )
        )

    ax.set_xlabel("Time to actuation (h)")
    ax.set_ylabel("Total chamber volume $V_{tot}$ (mL)")
    ax.set_title(
        f"Pareto front  |  {temperature} °C  |  "
        f"C = {config.MAP_CONCENTRATION_MM} mM (fixed)"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(config.OUTPUT_FOLDER, f"pareto_{temperature}C.png"),
        dpi=300
    )
    plt.close()
    print(f"  Saved pareto_{temperature}C.png")
