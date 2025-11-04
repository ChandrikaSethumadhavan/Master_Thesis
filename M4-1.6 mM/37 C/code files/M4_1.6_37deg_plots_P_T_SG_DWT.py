# ==========================================
# PRESSURE ANALYSIS SCRIPT WITH SG somoothing
# ==========================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error, r2_score
import pywt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import label
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# === File Paths ===
data_path = r"C:\Users\chand\Documents\GitHub\Thesis\M4-1.6 mM\37 C\raw dataset\concatinated\merged_pressure_log.csv"
plot_dir = r"C:\Users\chand\Documents\GitHub\Thesis\M4-1.6 mM\37 C\denoising"
os.makedirs(plot_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(data_path, parse_dates=["Datetime"])

# === Filter: upto certain amount of time if data is large===
df = df[df["Elapsed Time (hours)"] <= 70]

# === Extract Columns ===
time_raw = df["Elapsed Time (hours)"].values
abp2_raw = df["ABP2-Sch1 (kPa)"].values
mpr_raw = df["MPR-Sch2 (kPa)"].values
temperature = df["Temperature (C)"].values
humidity = df["Humidity (%RH)"].values

# === Interpolation to Uniform Time Grid ===
uniform_time = np.linspace(time_raw.min(), time_raw.max(),
                           int((time_raw.max() - time_raw.min()) * 3600))
abp2_interp = np.interp(uniform_time, time_raw, abp2_raw)
mpr_interp = np.interp(uniform_time, time_raw, mpr_raw)
temperature_interp = np.interp(uniform_time, time_raw, temperature)

# Smooth temperature to reduce noise (5 sec window)
smoothed_temp = uniform_filter1d(temperature_interp, size=300)

# === baseline: Auto Baseline Correction Based on Temp Stabilization ===
# === 1. Identify longest stable run with T > 36°C ===
binary_mask = temperature > 36.7
labeled, n_features = label(binary_mask)

if n_features == 0:
    raise ValueError("No region with temperature > 36°C found.")

# === 2. Require sustained stability for ≥ 15 min ===
sampling_rate_hz = len(uniform_time) / (uniform_time[-1] - uniform_time[0])  # samples per hour
min_duration_min = 10
min_points = int(min_duration_min * 60 * sampling_rate_hz / 3600)

lengths = [(i, np.sum(labeled == i)) for i in range(1, n_features + 1)]
long_segments = [seg for seg in lengths if seg[1] >= min_points]

if not long_segments:
    raise ValueError("No sufficiently long stable region found (≥15 min).")

# Pick the longest valid stable segment
longest_stable = max(long_segments, key=lambda x: x[1])[0]
stable_idx = np.where(labeled == longest_stable)[0]

# === 3. Start 20 minutes after stable region begins ===
offset_min = 20
offset_samples = int(offset_min * 60 * sampling_rate_hz / 3600)
start_idx = stable_idx[0] + offset_samples

# === 4. Safety checks ===
if start_idx >= len(uniform_time):
    raise IndexError("Start index after temperature stabilization exceeds data range.")

baseline_window_size = 60  # samples (e.g., 1 min if 1 Hz rate)
end_idx = min(start_idx + baseline_window_size, len(uniform_time))

if end_idx - start_idx < 60:
    raise ValueError("Baseline window too short after bounds check.")

# === 5. Compute baseline and correct ===
baseline_time_start = uniform_time[start_idx]
baseline_time_end = uniform_time[end_idx - 1]

# ABP2 baseline correction
baseline_pressure_abp2 = np.mean(abp2_interp[start_idx:end_idx])
abp2_corrected = abp2_interp - baseline_pressure_abp2

# MPR baseline correction
baseline_pressure_mpr = np.mean(mpr_interp[start_idx:end_idx])
mpr_corrected = mpr_interp - baseline_pressure_mpr

# === 5. Print diagnostic info ===
print(f"Baseline window: {baseline_time_start:.2f}–{baseline_time_end:.2f} h")
print(f"ABP2 baseline = {baseline_pressure_abp2:.4f} kPa")
print(f"MPR baseline  = {baseline_pressure_mpr:.4f} kPa")

# === Adaptive SG Window Selection ===
def compute_snr(signal, noise_region):
    noise_rms = np.sqrt(np.mean(noise_region**2))
    signal_p2p = np.ptp(signal)
    return 20 * np.log10(signal_p2p / noise_rms) if noise_rms != 0 else np.inf

def evaluate_sg(signal, min_win=11, max_win=99, step=2, polyorder=3):
    noise_region = signal[-36000:]
    results = []
    for w in range(min_win, max_win+1, step):
        if w >= len(signal): break
        smoothed = savgol_filter(signal, w, polyorder)
        snr = compute_snr(smoothed, noise_region)
        rmse = np.sqrt(mean_squared_error(signal, smoothed))
        results.append((w, snr, rmse))
    df = pd.DataFrame(results, columns=["window", "snr", "rmse"])
    df["snr_norm"] = (df["snr"] - df["snr"].min()) / (df["snr"].max() - df["snr"].min())
    df["rmse_norm"] = (df["rmse"].max() - df["rmse"]) / (df["rmse"].max() - df["rmse"].min())
    df["score"] = 0.6 * df["snr_norm"] + 0.4 * df["rmse_norm"]
    best_win = int(df.loc[df["score"].idxmax()]["window"])
    best_window_info = df.loc[df['score'].idxmax()]
    return best_win, best_window_info

abp2_opt_win, best_window_info = evaluate_sg(abp2_corrected)
abp2_sg = savgol_filter(abp2_corrected, abp2_opt_win, 3)

print(f"\nOptimal SG Window Size: {abp2_opt_win}\n")
print(best_window_info[['snr', 'rmse', 'score']])

mpr_opt_win, best_window_info_mpr = evaluate_sg(mpr_corrected)
mpr_sg = savgol_filter(mpr_corrected, mpr_opt_win, 3)

print(f"\n[MPR] Optimal SG Window Size: {mpr_opt_win}")
print(best_window_info_mpr[['snr', 'rmse', 'score']])

# === Save Cleaned Data ===
# Export baseline-corrected, smoothed signal, to save the cleaned version for modeling:
clean_df = pd.DataFrame({
    "Elapsed Time (h)": uniform_time,
    "ABP2 Raw (baseline-corrected, kPa)": abp2_corrected,
    "ABP2 SG Smoothed (kPa)": abp2_sg,
    "MPR Raw (baseline-corrected, kPa)": mpr_corrected,
    "MPR SG Smoothed (kPa)": mpr_sg,
    "Temperature (°C)": temperature_interp
})


# === for plotting - Crop data from baseline_time_start onward ===
mask_after_baseline = uniform_time >= baseline_time_start
plot_time = uniform_time[mask_after_baseline]
plot_abp2_interp = abp2_interp[mask_after_baseline]
plot_abp2_corrected = abp2_corrected[mask_after_baseline]
plot_mpr_interp = mpr_interp[mask_after_baseline]
plot_mpr_corrected = mpr_corrected[mask_after_baseline]
plot_temperature = smoothed_temp[mask_after_baseline]

plot_abp2_sg = abp2_sg[mask_after_baseline]
plot_mpr_sg = mpr_sg[mask_after_baseline]

# === Plotting ===
# aesthetics & fonts
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
dpi = 900

# === 1: Plot ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- Pressure on left y-axis ---
ax1.plot(uniform_time, abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
ax1.plot(uniform_time, abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)

ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.set_ylim(np.min(abp2_corrected), np.max(abp2_interp)+1.5 )
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Combine legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)
# ax1.set_title("ABP2 Thermal Correction with Temperature Overlay", fontsize=16)

# plt.title("ABP2 Thermal Correction with Temperature Overlay", fontsize=13)
plt.tight_layout()

# --- Inset: First 15 minutes ---
inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
                      bbox_to_anchor=(-0.12, 0.2, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_mask = (uniform_time <= 5)
inset_ax.plot(uniform_time[inset_mask], abp2_corrected[inset_mask],
              color="#921b66db", lw=1.5, label="ABP2 Corrected")
inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 5 hours", fontsize=10)

# --- Temperature in inset (right y-axis) ---
inset_ax2 = inset_ax.twinx()
inset_ax2.plot(uniform_time[inset_mask], temperature_interp[inset_mask],
               color="tab:red", lw=1.2, alpha=0.6)
inset_ax2.set_ylabel("Temp (°C)", fontsize=8)
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=8, labelcolor="black")

plt.savefig(os.path.join(plot_dir, "1_M4_37_ABP2_thermal_overlay_with_inset.png"), dpi=dpi)






# === 1: Plot ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- Pressure on left y-axis ---
ax1.plot(uniform_time, mpr_interp, label="MPR Raw", color="#1f78b4", lw=2.5)
ax1.plot(uniform_time, mpr_corrected, label="MPR Corrected", color="#921b66db", lw=2.5)

ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.set_ylim(np.min(mpr_corrected), np.max(mpr_interp)+1.5 )
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Combine legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)
ax1.set_title("MPR Thermal Correction with Temperature Overlay", fontsize=16)


plt.tight_layout()

# --- Inset: First 15 minutes ---
inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
                      bbox_to_anchor=(-0.12, 0.2, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_mask = (uniform_time <= 5)
inset_ax.plot(uniform_time[inset_mask], abp2_corrected[inset_mask],
              color="#921b66db", lw=1.5, label="ABP2 Corrected")
inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 5 hours", fontsize=10)

# --- Temperature in inset (right y-axis) ---
inset_ax2 = inset_ax.twinx()
inset_ax2.plot(uniform_time[inset_mask], temperature_interp[inset_mask],
               color="tab:red", lw=1.2, alpha=0.6)
inset_ax2.set_ylabel("Temp (°C)", fontsize=8)
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=8, labelcolor="black")

plt.savefig(os.path.join(plot_dir, "1b_M4_37_MPR_thermal_overlay_with_inset.png"), dpi=dpi)













# === 2: plot: Starting from Baseline (T > 36 °C Stabilized)

fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- Pressure on left y-axis ---
# ax1.plot(plot_time, plot_abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
ax1.plot(plot_time, plot_abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)

ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.set_ylim(np.min(plot_abp2_corrected) - 0.5, np.max(plot_abp2_corrected) + 1.5)
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)

ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Combine legend only once ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)
# ax1.set_title("Baseline Corrected ABP2 Pressure with Temperature Overlay", fontsize=16)
plt.tight_layout()

# --- Inset: First 5 hours after baseline ---
inset_mask = plot_time <= (plot_time[0] + 5)
inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
                      bbox_to_anchor=(-0.15, 0.2, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_ax.plot(plot_time[inset_mask], plot_abp2_corrected[inset_mask],
              color="#921b66db", lw=2.5, label="ABP2 Corrected")
inset_ax.set_ylim(np.min(plot_abp2_corrected[inset_mask]) - 0.1, np.max(plot_abp2_corrected[inset_mask]) + 0.5)
inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)

inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 3 h after baseline", fontsize=10)

inset_ax2 = inset_ax.twinx()
inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
               color="tab:red", lw=1.5, alpha=0.6)
inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
inset_ax2.set_ylabel("Temp (°C)", fontsize=9)
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=8, labelcolor="black")

# === Save cropped plot ===
plt.savefig(os.path.join(plot_dir, "2_M4_37_ABP2_Cropped_from_Baseline.png"), dpi=dpi)


#MPR 

fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- Pressure on left y-axis ---
# ax1.plot(plot_time, plot_abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
ax1.plot(plot_time, plot_mpr_corrected, label="MPR Corrected", color="#921b66db", lw=2.5)

ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.set_ylim(np.min(plot_mpr_corrected) - 0.5, np.max(plot_mpr_corrected) + 1.5)
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)

ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Combine legend only once ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)
# ax1.set_title("Baseline Corrected ABP2 Pressure with Temperature Overlay", fontsize=16)
plt.tight_layout()

# --- Inset: First 5 hours after baseline ---
inset_mask = plot_time <= (plot_time[0] + 5)
inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
                      bbox_to_anchor=(-0.15, 0.2, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_ax.plot(plot_time[inset_mask], plot_mpr_corrected[inset_mask],
              color="#921b66db", lw=2.5, label="MPR Corrected")
inset_ax.set_ylim(np.min(plot_mpr_corrected[inset_mask]) - 0.1, np.max(plot_mpr_corrected[inset_mask]) + 0.5)
inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)

inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 3 h after baseline", fontsize=10)

inset_ax2 = inset_ax.twinx()
inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
               color="tab:red", lw=1.5, alpha=0.6)
inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
inset_ax2.set_ylabel("Temp (°C)", fontsize=9)
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=8, labelcolor="black")

# === Save cropped plot ===
plt.savefig(os.path.join(plot_dir, "2b_M4_37_MPR_Cropped_from_Baseline.png"), dpi=dpi)


























# === Plot 3: SG-Smoothed Pressure after Baseline ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- Pressure on left y-axis ---
ax1.plot(plot_time, plot_abp2_sg, label="SG smoothed pressure signal", color="#921b66db", lw=2.5)

ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.set_ylim(np.min(plot_abp2_sg) - 0.5, np.max(plot_abp2_sg) + 1.5)
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelcolor="black", labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(plot_time, plot_temperature, color="tab:red", lw=2, alpha=0.6, label="Temperature profile")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)

plt.tight_layout()

# --- Inset: First 5 hours ---
inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
                      bbox_to_anchor=(-0.18, 0.2, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_ax.plot(plot_time[inset_mask], plot_abp2_sg[inset_mask],
              color="#921b66db", lw=2.5, label="ABP2 SG Smoothed")
inset_ax.set_ylim(np.min(plot_abp2_sg[inset_mask]) - 0.1, np.max(plot_abp2_sg[inset_mask]) + 0.5)
inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 5 h", fontsize=12)

inset_ax2 = inset_ax.twinx()
inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
               color="tab:red", lw=2, alpha=0.6)
inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
inset_ax2.set_ylabel("Temp (°C)", fontsize=12)
inset_ax2.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=9, labelcolor="black")

plt.savefig(os.path.join(plot_dir, "3_M4_37_ABP2_SG_Smoothed.png"), dpi=dpi)



# === Plot 4: MPR SG-Smoothed Pressure after Baseline ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- Pressure on left y-axis ---
ax1.plot(plot_time, plot_mpr_sg, label="SG smoothed pressure signal", color="#921b66db", lw=2.5)

ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.set_ylim(np.min(plot_mpr_sg) - 0.5, np.max(plot_mpr_sg) + 1.5)
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelcolor="black", labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(plot_time, plot_temperature, color="tab:red", lw=2, alpha=0.6, label="Temperature profile")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)

plt.tight_layout()

# --- Inset: First 5 hours ---
inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
                      bbox_to_anchor=(-0.18, 0.2, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_ax.plot(plot_time[inset_mask], plot_mpr_sg[inset_mask],
              color="#921b66db", lw=2.5, label="ABP2 SG Smoothed")
inset_ax.set_ylim(np.min(plot_mpr_sg[inset_mask]) - 0.1, np.max(plot_mpr_sg[inset_mask]) + 0.5)
inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 5 h", fontsize=12)

inset_ax2 = inset_ax.twinx()
inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
               color="tab:red", lw=2, alpha=0.6)
inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
inset_ax2.set_ylabel("Temp (°C)", fontsize=12)
inset_ax2.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=9, labelcolor="black")

plt.savefig(os.path.join(plot_dir, "3_M4_37_MPR_SG_Smoothed.png"), dpi=dpi)




























# === Plot 5: First-order fit to SG-smoothed pressure ===

# --- First-order kinetic model ---
def first_order_model(t, Pmax, k):
    return Pmax * (1 - np.exp(-k * t))

# Fit to SG-smoothed pressure
popt, _ = curve_fit(first_order_model, plot_time, plot_abp2_sg, p0=[np.max(plot_abp2_sg), 0.1])
fitted_model = first_order_model(plot_time, *popt)

# Metrics
r2 = r2_score(plot_abp2_sg, fitted_model)
rmse = np.sqrt(mean_squared_error(plot_abp2_sg, fitted_model))
n = len(plot_time)
k_params = len(popt)
residuals = plot_abp2_sg - fitted_model
sse = np.sum(residuals**2)
aic = n * np.log(sse / n) + 2 * k_params

print(f"[First-Order Fit] R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(5, 4.5), dpi=dpi)
ax.plot(plot_time, plot_abp2_sg, color="#921b66db", lw=3, alpha=0.7, label=fr"S-G smoothed pressure signal" + "\n")
ax.plot(plot_time, fitted_model, linestyle="--", color="#009F82", lw=2,
        label=fr"First-order fit ($R^2$ = {r2:.3f})" + "\n" + r"$P(t) = P_{\max}(1 - e^{-kt})$" + "\n")

ax.set_xlabel("Elapsed Time (h)", fontsize=20)
ax.set_ylabel("Pressure (kPa)", fontsize=20)
ax.set_ylim(np.min(plot_abp2_sg) - 0.5, np.max(plot_abp2_sg) + 0.5)
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax.minorticks_on()
ax.legend(fontsize=14, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "5_M4_37_FirstOrder_Fit_SG.png"), dpi=dpi)




# === Plot 5: First-order fit to SG-smoothed pressure on MPR ===

# --- First-order kinetic model ---
def first_order_model(t, Pmax, k):
    return Pmax * (1 - np.exp(-k * t))

# Fit to SG-smoothed pressure
popt, _ = curve_fit(first_order_model, plot_time, plot_mpr_sg, p0=[np.max(plot_mpr_sg), 0.1])
fitted_model = first_order_model(plot_time, *popt)

# Metrics
r2 = r2_score(plot_mpr_sg, fitted_model)
rmse = np.sqrt(mean_squared_error(plot_mpr_sg, fitted_model))
n = len(plot_time)
k_params = len(popt)
residuals = plot_mpr_sg - fitted_model
sse = np.sum(residuals**2)
aic = n * np.log(sse / n) + 2 * k_params

print(f"[First-Order Fit] R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(5, 4.5), dpi=dpi)
ax.plot(plot_time, plot_mpr_sg, color="#921b66db", lw=3, alpha=0.7, label=fr"S-G smoothed pressure signal" + "\n")
ax.plot(plot_time, fitted_model, linestyle="--", color="#009F82", lw=2,
        label=fr"First-order fit ($R^2$ = {r2:.3f})" + "\n" + r"$P(t) = P_{\max}(1 - e^{-kt})$" + "\n")

ax.set_xlabel("Elapsed Time (h)", fontsize=20)
ax.set_ylabel("Pressure (kPa)", fontsize=20)
ax.set_ylim(np.min(plot_mpr_sg) - 0.5, np.max(plot_mpr_sg) + 0.5)
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax.minorticks_on()
ax.legend(fontsize=14, loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "5_M4_37__mpr_FirstOrder_Fit_SG.png"), dpi=dpi)






















# === Plot 6: Oxygen Release (µmol) from Pressure ===

# Constants
volume_solution_L = 0.006 # 6ml
concentration_mM = 1.6 # 1.6 mM
V_headspace_L = 0.004  # 4 ml = 0.004 L 
R = 0.082057  # L·atm/mol·K
ratio = volume_solution_L / V_headspace_L

max_o2_umol = concentration_mM * volume_solution_L * 1e3 # micromol

pressure_atm = plot_abp2_sg / 101.325      # Convert pressure (kPa) to atm 
pressure_atm_mpr = plot_mpr_sg / 101.325   # MPR absolute pressure SG smoothed data 
temperature_K = plot_temperature + 273.15   # temperature to Kelvin

# Compute µmol O₂
o2_micromol = ((pressure_atm * V_headspace_L) / (R * temperature_K)) * 1e6
o2_micromol_mpr = ((pressure_atm_mpr * V_headspace_L) / (R * temperature_K)) * 1e6


# === Efficiency factor calculation (η) up to 60 hours ===
efficiency_limit_h = 60

# Mask data up to that time
mask_eff = plot_time <= efficiency_limit_h

# Compute max O₂ released within 60 h
max_o2_abp2_60h = np.max(o2_micromol[mask_eff])
max_o2_mpr_60h = np.max(o2_micromol_mpr[mask_eff])

# Compute efficiency relative to theoretical O₂ capacity
eta_abp2 = max_o2_abp2_60h / max_o2_umol
eta_mpr = max_o2_mpr_60h / max_o2_umol

print(f"\n=== Efficiency Factors (up to {efficiency_limit_h} h) ===")
print(f"ABP2 η = {eta_abp2:.3f}  →  {eta_abp2*100:.1f}% of theoretical release")
print(f"MPR  η = {eta_mpr:.3f}  →  {eta_mpr*100:.1f}% of theoretical release")

# Fit O₂ release to first-order model
def o2_model(t, N_max, k):
    return N_max * (1 - np.exp(-k * t))

# abp2
t_fit = plot_time - plot_time[0]    # Shift time so fit starts from zero
popt_o2, _ = curve_fit(o2_model, t_fit, o2_micromol, p0=[np.max(o2_micromol), 0.02])
fitted_o2 = o2_model(t_fit, *popt_o2)
r2_o2 = r2_score(o2_micromol, fitted_o2)

# mpr
t_fit = plot_time - plot_time[0]    # Shift time so fit starts from zero
popt_o2_mpr, _ = curve_fit(o2_model, t_fit, o2_micromol_mpr, p0=[np.max(o2_micromol_mpr), 0.02])
fitted_o2_mpr = o2_model(t_fit, *popt_o2_mpr)
r2_o2_mpr = r2_score(o2_micromol_mpr, fitted_o2_mpr)

df_o2_export = pd.DataFrame({
    "Sample": "1.6 mM @ 37deg",
    "Time (h)": plot_time,
    "ABP2 SG smoothed pressure (kPa)": plot_abp2_sg,
    "MPR SG smoothed pressure (kPa)": plot_mpr_sg,
    "calibrated temperature (K)": plot_temperature,
    "ABP2 O2 Released (µmol)": o2_micromol,
    "MPR O2 Released (µmol)": o2_micromol_mpr,
    "ABP2 % Conversion": (o2_micromol / max_o2_umol) * 100,
    "MPR % Conversion": (o2_micromol_mpr / max_o2_umol) * 100,
    "Soln/Headspace Ratio": ratio,
    "Max O2 Possible (µmol)": max_o2_umol,

})

df_o2_export.to_csv("M4_1.6mM_37deg_pressure_o2_release.csv", index=False)

# === Plot ABP2 ===
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)
ax.plot(plot_time, o2_micromol, label="O2 released (µmol)", lw=2.5, color="#1f77b4")
ax.plot(plot_time, fitted_o2, '--', lw=2, color="#ff7f0e", label=fr"First-order fit ($R^2$ = {r2_o2:.3f})")
# ax.axhline(o2_theoretical, linestyle='--', color='gray', linewidth=1.5,
#            label=f"Theoretical max: {o2_theoretical:.2f} µmol")

ax.set_xlabel("Elapsed Time (h)", fontsize=18)
ax.set_ylabel("O2 released (µmol)", fontsize=18)
ax.set_ylim(bottom=0)
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax.minorticks_on()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=13, frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "6_M4_37_O2_ABP2_release_from_pressure.png"), dpi=dpi)

# === Plot MPR ===
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)
ax.plot(plot_time, o2_micromol_mpr, label="O2 released (µmol, MPR)", lw=2.5, color="#1b9e77")
ax.plot(plot_time, fitted_o2_mpr, '--', lw=2, color="#ff7f0e",
        label=fr"First-order fit ($R^2$ = {r2_o2_mpr:.3f})")

ax.set_xlabel("Elapsed Time (h)", fontsize=18)
ax.set_ylabel("O2 released (µmol)", fontsize=18)
ax.set_ylim(bottom=0)
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax.minorticks_on()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=13, frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "6_M4_37_O2_MPR_release_from_pressure.png"), dpi=dpi)

# === IEEE I2MTC submission plots ===

# === Plot 7: O2 Release (µmol) with Temperature and First-Order Fit (ABP2) ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- O2 release on left y-axis ---
ax1.plot(plot_time, o2_micromol, label="$O_2$ released (µmol)", color="#1f77b4", lw=2.75)
ax1.plot(plot_time, fitted_o2, '--', color="#ff7f0e", lw=2.25,
         label=fr"First-order fit ($R^2$ = {r2_o2:.3f})")

ax1.set_xlabel("Elapsed Time (h)", fontsize=22)
ax1.set_ylabel("$O_2$ released (µmol)", fontsize=22, color="black")
ax1.set_ylim(np.min(o2_micromol) - 0.5, np.max(o2_micromol) + 1.5)
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)

ax1.tick_params(axis='y', labelcolor="black", labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=14)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1.5, alpha=0.6, label="Temperature profile")
ax2.set_ylabel("Temperature (°C)", fontsize=22, color="black")
ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.05)
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=14)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Combine legends ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)

plt.tight_layout()

# --- Inset: First 5 hours ---
inset_mask = plot_time <= (plot_time[0] + 5)
inset_ax = inset_axes(ax1, width="35%", height="35%", loc="lower right",
                      bbox_to_anchor=(-0.18, 0.275, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_ax.plot(plot_time[inset_mask], o2_micromol[inset_mask],
              color="#1f77b4", lw=2.75, label="$O_2$ released")
inset_ax.plot(plot_time[inset_mask], fitted_o2[inset_mask],
              '--', color="#ff7f0e", lw=2.0, label="First-order fit")

inset_ax.set_ylim(np.min(o2_micromol[inset_mask]) - 0.1, np.max(o2_micromol[inset_mask]) + 0.5)
inset_ax.set_ylabel("$O_2$ (µmol)", fontsize=12)
inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 5 h", fontsize=12)

# --- Temperature inset ---
inset_ax2 = inset_ax.twinx()
inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
               color="tab:red", lw=2, alpha=0.6)
inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.05)
inset_ax2.set_ylabel("Temp (°C)", fontsize=12)
inset_ax2.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=9, labelcolor="black")

plt.savefig(os.path.join(plot_dir, "7_M4_37_ABP2_O2_with_Temperature_and_Fit.png"), dpi=dpi)

# === Plot 7: O2 Release (µmol) with Temperature and First-Order Fit (MPR) ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- O2 release on left y-axis ---
ax1.plot(plot_time, o2_micromol_mpr, label="$O_2$ released (µmol)", color="#1b9e77", lw=2.75)
ax1.plot(plot_time, fitted_o2_mpr, '--', color="#ff7f0e", lw=2.25,
         label=fr"First-order fit ($R^2$ = {r2_o2_mpr:.3f})")

ax1.set_xlabel("Elapsed Time (h)", fontsize=22)
ax1.set_ylabel("$O_2$ released (µmol)", fontsize=22, color="black")
ax1.set_ylim(np.min(o2_micromol_mpr) - 0.5, np.max(o2_micromol_mpr) + 1.5)
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)

ax1.tick_params(axis='y', labelcolor="black", labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=14)
ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax1.minorticks_on()

# --- Temperature on right y-axis ---
ax2 = ax1.twinx()
ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1.5, alpha=0.6, label="Temperature profile")
ax2.set_ylabel("Temperature (°C)", fontsize=22, color="black")
ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.05)
ax2.grid(False)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=14)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

# --- Combine legends ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)

plt.tight_layout()

# --- Inset: First 5 hours ---
inset_mask = plot_time <= (plot_time[0] + 5)
inset_ax = inset_axes(ax1, width="35%", height="35%", loc="lower right",
                      bbox_to_anchor=(-0.18, 0.275, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_ax.plot(plot_time[inset_mask], o2_micromol_mpr[inset_mask],
              color="#1b9e77", lw=2.75, label="$O_2$ released")
inset_ax.plot(plot_time[inset_mask], fitted_o2_mpr[inset_mask],
              '--', color="#ff7f0e", lw=2.0, label="First-order fit")

inset_ax.set_ylim(np.min(o2_micromol_mpr[inset_mask]) - 0.1, np.max(o2_micromol_mpr[inset_mask]) + 0.5)
inset_ax.set_ylabel("$O_2$ (µmol)", fontsize=12)
inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=12)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 5 h", fontsize=12)

# --- Temperature inset ---
inset_ax2 = inset_ax.twinx()
inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
               color="tab:red", lw=2, alpha=0.6)
inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.05)
inset_ax2.set_ylabel("Temp (°C)", fontsize=12)
inset_ax2.yaxis.set_major_locator(MultipleLocator(0.5))
inset_ax2.grid(False)
inset_ax2.tick_params(labelsize=9, labelcolor="black")

plt.savefig(os.path.join(plot_dir, "7_M4_37_MPR_O2_with_Temperature_and_Fit.png"), dpi=dpi)
