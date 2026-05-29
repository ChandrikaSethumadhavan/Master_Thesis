import os
import numpy as np
import matplotlib.pyplot as plt
import config


def plot_pressure_map(result):
    X, Y = np.meshgrid(result["vg_vals"], result["vsol_vals"])

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, result["P_map"], levels=25)
    plt.colorbar(contour, label="ΔPmax (kPa)")
    plt.xlabel("Vg (mL)")
    plt.ylabel("Vsol (mL)")
    plt.title(
        f"Pressure Design Map | T={result['temperature_c']} °C | "
        f"C={result['C_target_mM']} mM | Sensor={result['sensor']}"
    )
    plt.tight_layout()

    fname = os.path.join(config.OUTPUT_FOLDER, f"pressure_map_{int(result['temperature_c'])}C.png")
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_feasibility_map(result):
    X, Y = np.meshgrid(result["vg_vals"], result["vsol_vals"])

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, result["feasible_map"], levels=[-0.1, 0.5, 1.1])
    threshold = plt.contour(X, Y, result["P_map"], levels=[config.PACT_KPA], linewidths=2)
    plt.clabel(threshold, fmt={config.PACT_KPA: f"{config.PACT_KPA:.1f} kPa"})
    plt.xlabel("Vg (mL)")
    plt.ylabel("Vsol (mL)")
    plt.title(
        f"Actuation Feasibility | T={result['temperature_c']} °C | "
        f"Pact={config.PACT_KPA} kPa"
    )
    plt.tight_layout()

    fname = os.path.join(config.OUTPUT_FOLDER, f"feasibility_map_{int(result['temperature_c'])}C.png")
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_time_map(result):
    X, Y = np.meshgrid(result["vg_vals"], result["vsol_vals"])
    TMAP = result["t_act_map"].copy()

    plt.figure(figsize=(8, 6))
    masked = np.ma.masked_invalid(TMAP)
    contour = plt.contourf(X, Y, masked, levels=25)
    plt.colorbar(contour, label="Time to actuation (h)")
    plt.xlabel("Vg (mL)")
    plt.ylabel("Vsol (mL)")
    plt.title(
        f"Time to Actuation | T={result['temperature_c']} °C | "
        f"k(GPR)={result['k_hat']:.5f} 1/h"
    )
    plt.tight_layout()

    fname = os.path.join(config.OUTPUT_FOLDER, f"time_to_actuation_{int(result['temperature_c'])}C.png")
    plt.savefig(fname, dpi=300)
    plt.close()


def save_all_plots(results):
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    for result in results:
        plot_pressure_map(result)
        plot_feasibility_map(result)
        plot_time_map(result)