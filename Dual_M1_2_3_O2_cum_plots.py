import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats
import matplotlib.ticker as ticker
from scipy.stats import linregress
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset



# Define optical sample data, M1 = 20 mmicromolar
samples = {
    "M1": {
        "time": np.array([0, 0.12, 1.12, 2.12, 3.09, 4.12, 5.12, 5.24, 6.1, 27.85, 48.55, 
                          71.5, 143.76, 147.75, 168.34, 171.87, 239.3, 311.9, 334.85, 407.14, 
                          455.5, 479.67, 576.4, 598.6]), # discrete time points in hours
        "C_ANT": np.array([0, 0.048, 0.423, 0.806, 1.162, 1.355, 1.887, 1.929, 2.228, 8.271, 
                           12.154, 14.966, 18.758, 18.85, 19.23, 19.282, 19.805, 19.952, 19.969, 
                           19.99, 19.997, 19.998, 19.99972, 19.9998]),
        "A_400": np.array([0, 0.00013, 0.0018, 0.0041, 0.0065, 0.0094, 0.012, 0.0125, 0.0146, 
                           0.053, 0.0666, 0.0778, 0.0974, 0.0979, 0.1010, 0.1014, 0.1069, 0.109, 
                           0.109, 0.108, 0.1079, 0.1079, 0.1081, 0.1084]),                      # A400 corrected from the setup
        "A_uvvis": [0, 0.001, 0.002, 0.005, 0.006, 0.009, 0.011, 0.014, 0.017, 0.052, 0.078, 
                    0.10, 0.137, 0.138, 0.144, 0.145, 0.154, 0.158, 0.159, 0.160, 0.161, 0.160, 
                    0.161, 0.160],                                                              # A400UV-vis (from spectrometer))
        "epsilon_setup": 5420.5,    # Molar absorptivity - calculated from A400 final value representing full ANT (from beer lambert law, it is customised to individual components)
        "epsilon_uvvis": 8000,      # Molar absorptivity - calculated from A400UV-vis final value representing full ANT (same as above)
        "volume": 0.0026553,
        "marker": 'o'
    }
}

### Function to compute O2 release from absorbance
# def compute_o2_from_absorbance(A, epsilon, volume):    #Function to convert absorbance readings into moles of O₂ released
#     C_ANT = np.array(A) / epsilon * 1e6  # µM
#     return C_ANT * volume  # µmol                           #has been skipped anyway

# === Load Pressure Data ===
pressure_files = {

    "M2": "M2_2.18mM_pressure_o2_release.csv"

}
pressure_data = {
    k: pd.read_csv(v) for k, v in pressure_files.items()
}

# === Compare ===
fig_abs, ax_abs = plt.subplots(figsize=(7, 5))     #Creates two empty plot windows
fig_ratio, ax_ratio = plt.subplots(figsize=(7, 5))

# --- Optical Data ---
for name, s in samples.items():
    # Mask for time ≤ 250 h
    mask = s["time"] <= 250  # Creates array of True/False values, If time = [0, 100, 200, 300, 400], then mask = [True, True, True, False, False]
    time_limited = s["time"][mask]  #  # Only keeps elements where mask=True
    o2_limited = s["C_ANT"][mask] * s["volume"] # Only keeps elements where mask = True and also Convert to µmol
    max_o2 = s["volume"] * max(s["C_ANT"])  # Single number representing maximum O₂ ever reached

    ax_abs.plot(time_limited, o2_limited, label=f"{name} (optical)", marker=s["marker"])
    ax_ratio.scatter(s["volume"] / 0.0008, max(o2_limited), label=f"{name} (optical)", marker=s["marker"])  # 0.8 ml headspace

# /* ^^^^^ for example:  Solution volume = 2.655 ml = 0.002655 L
# Headspace volume = 0.8 ml = 0.0008 L
# Volume ratio = 0.002655 ÷ 0.0008 = 3.32

# --- Pressure Data ---
for label, df in pressure_data.items():
    # Time conversion: seconds → hours
    t = df["Time (s)"]
    o2 = df["O2 Released (µmol)"]
    max_o2 = df["Max O2 Possible (µmol)"].iloc[0] #(this value is the same for all rows, so just grab the first one)
    ratio = df["Soln/Headspace Ratio"].iloc[0]  #(this value is the same for all rows, so just grab the first one)

    marker = 'x' if "37" in label else 'D'
    label_fmt = f"{label} (pressure)"
    ax_abs.plot(t, o2, label=label_fmt, marker=marker, linewidth=2.5)
    ax_ratio.scatter(ratio, np.max(o2), label=label_fmt, marker=marker)

# === Final Plot Formatting ===
for ax, ylabel in zip([ax_abs], ["O₂ Released (µmol)"]):
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

ax_ratio.set_xlabel("Solution/Headspace Volume Ratio")
ax_ratio.set_ylabel("Max O₂ Released (µmol)")
ax_ratio.legend()
ax_ratio.grid(True)
plt.tight_layout()
plt.show()

# ====================================================

# === Initial moles of ANT-EPO (1:1 conversion to O2) ===
init_o2_optical = {"M1": 20 * 2.655 / 1000, "M2": 25 * 2.743 / 1000, "M3": 30 * 2.766 / 1000}  # µmol
init_o2_pressure = {"M1": 2.5 * 6, "M2": 2.18 * 6}  # µmol

# === Normalized % Conversion Plot ===
fig_conv, ax_conv = plt.subplots(figsize=(7, 5))

# --- Optical ---
for name, s in samples.items():
    # Mask for time ≤ 250 h
    mask = s["time"] <= 250
    time_limited = s["time"][mask]
    o2_limited = s["C_ANT"][mask] * s["volume"]
    norm_percent = (o2_limited / init_o2_optical[name]) * 100
    ax_conv.plot(time_limited, norm_percent, label=f"{name} (optical)", marker=s["marker"], linestyle='-')

# --- Pressure ---
for label, df in pressure_data.items():
    t_hr = df["Time (s)"]
    o2 = df["O2 Released (µmol)"]
    label_key = "M2" if "37" in label else label
    norm_percent = (o2 / init_o2_pressure[label_key]) * 100
    marker = 'x' if "37" in label else 'D'
    ax_conv.plot(t_hr, norm_percent, label=f"{label} (pressure)", marker=marker, linestyle='--')

# === Format Plot ===
ax_conv.set_xlabel("Time (hours)")
ax_conv.set_ylabel("O₂ Released (% conversion)")
ax_conv.set_ylim(0, 110)
ax_conv.legend()
ax_conv.grid(True)
ax_conv.set_title("Normalized O₂ Release Across Systems")

# === Show Plots ===
plt.tight_layout()
plt.show()
