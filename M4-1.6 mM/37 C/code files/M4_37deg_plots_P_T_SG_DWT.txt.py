
# ==========================================
# PRESSURE ANALYSIS SCRIPT WITH SG somoothing and DWT
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
data_path =  r"C:\Users\chand\Documents\GitHub\Thesis\M4-1.6 mM\37 C\data files\sensor_drift_dropped.csv"
plot_dir = r"C:\Users\chand\Documents\GitHub\Thesis\M4-1.6 mM\37 C\plots"
os.makedirs(plot_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(data_path, parse_dates=["Datetime"])

# === Extract Columns ===
time_raw = df["Elapsed Time (hours)"].values
abp2_raw = df["ABP2-Sch1 (kPa)"].values
temperature = df["Temperature (C)"].values

# === Interpolation to Uniform Time Grid ===
uniform_time = np.linspace(time_raw.min(), time_raw.max(), int((time_raw.max() - time_raw.min()) * 3600))
abp2_interp = np.interp(uniform_time, time_raw, abp2_raw)
temperature_interp = np.interp(uniform_time, time_raw, temperature)


# === Improved Baseline Correction: first 1.5 h average ===
baseline_duration_h = 1.5  # hours
baseline_mask = uniform_time <= baseline_duration_h

if np.any(baseline_mask):
    baseline_avg = np.mean(abp2_interp[baseline_mask])
else:
    baseline_avg = np.mean(abp2_interp[:int(1.5 * 3600)])  # fallback for uniform grid

# Shift signal so that it starts at 0 kPa
abp2_corrected = abp2_interp - baseline_avg
abp2_corrected -= abp2_corrected[0]  # ensure first point = 0

print(f"[Baseline] Averaged over first {baseline_duration_h} hours")
print(f"[Baseline] Mean baseline pressure: {baseline_avg:.4f} kPa")
print(f"[Baseline] First corrected value: {abp2_corrected[0]:.4f} kPa")


# === Adaptive SG Window Selection ===
def compute_snr(signal, noise_region):
    noise_rms = np.sqrt(np.mean(noise_region**2))
    signal_p2p = np.ptp(signal)
    return 20 * np.log10(signal_p2p / noise_rms) if noise_rms != 0 else np.inf

def evaluate_sg(signal, min_win=13, max_win=99, step=2, polyorder=3):
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

# === DWT Denoising ===
def dwt_denoise(signal, keep_levels=(3, 4, 5), wavelet="db4"):
    coeffs = pywt.wavedec(signal, wavelet, level=pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len))
    for i in range(1, len(coeffs)):
        if i not in keep_levels:
            coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

abp2_dwt = dwt_denoise(abp2_sg)

# === Save Cleaned Data ===
clean_df = pd.DataFrame({
    "Elapsed Time (h)": uniform_time,
    "ABP2 Raw (baseline-corrected, kPa)": abp2_corrected,
    "ABP2 SG Smoothed (kPa)": abp2_sg,
    "ABP2 SG + DWT Denoised (kPa)": abp2_dwt,
    "Temperature (°C)": temperature_interp
})

# === Plotting Settings ===
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
dpi = 900

# === Plot 1: Raw + Corrected ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)
ax1.plot(uniform_time, abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
ax1.plot(uniform_time, abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)
ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20)
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax2 = ax1.twinx()
ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20)
ax1.legend(fontsize=10, loc="lower right", frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "1_ABP2_raw_vs_corrected.png"), dpi=dpi)


# === Plot: ABP2 Corrected Pressure + Temperature with Inset ===
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator

fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# --- Pressure (left y-axis) ---
ax1.plot(uniform_time, abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)
ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelcolor="black", labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.minorticks_on()

# --- Temperature (right y-axis) ---
ax2 = ax1.twinx()
ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1.2, alpha=0.8, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.tick_params(axis='y', labelcolor="black", labelsize=12)
ax2.minorticks_on()

# --- Combine legends ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)

# --- Inset: First 3 hours ---
inset_ax = inset_axes(ax1, width="38%", height="38%", loc="lower right",
                      bbox_to_anchor=(-0.15, 0.22, 1, 1),
                      bbox_transform=ax1.transAxes, borderpad=0)

inset_mask = uniform_time <= 5
inset_ax.plot(uniform_time[inset_mask], abp2_corrected[inset_mask],
              color="#921b66db", lw=1.8)
inset_ax.set_ylabel("Pressure (kPa)", fontsize=10)
inset_ax.xaxis.set_major_locator(MultipleLocator(1))
inset_ax.tick_params(labelsize=10)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.set_title("First 3 h after baseline", fontsize=9)

inset_ax2 = inset_ax.twinx()
inset_ax2.plot(uniform_time[inset_mask], temperature_interp[inset_mask],
               color="tab:red", lw=1, alpha=0.8)
inset_ax2.set_ylabel("Temp (°C)", fontsize=8)
inset_ax2.tick_params(labelsize=8, labelcolor="black")
inset_ax2.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "ABP2_Temperature_overlay.png"), dpi=dpi)
plt.show()


# === Plot 2: SG Smoothed Pressure ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)
ax1.plot(uniform_time, abp2_sg, label="SG Smoothed Pressure", color="#921b66db", lw=2.5)
ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20)
ax2 = ax1.twinx()
ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1.2, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "2_ABP2_SG_smoothed.png"), dpi=dpi)

# === Plot 3: SG + DWT Denoised ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)
ax1.plot(uniform_time, abp2_dwt, label="SG + DWT Denoised", color="#009E73", lw=2.5)
ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20)
ax2 = ax1.twinx()
ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "3_ABP2_SG_DWT.png"), dpi=dpi)

# === Plot 4: First-order Fit (using SG smoothed) ===
def first_order_model(t, Pmax, k):
    return Pmax * (1 - np.exp(-k * t))

popt, _ = curve_fit(first_order_model, uniform_time, abp2_sg, p0=[np.max(abp2_sg), 0.1])
fitted_model = first_order_model(uniform_time, *popt)
r2 = r2_score(abp2_sg, fitted_model)
rmse = np.sqrt(mean_squared_error(abp2_sg, fitted_model))
n = len(uniform_time)
k_params = len(popt)
sse = np.sum((abp2_sg - fitted_model)**2)
aic = n * np.log(sse / n) + 2 * k_params

print(f"[First-Order Fit] R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")

fig, ax = plt.subplots(figsize=(5, 4.5), dpi=dpi)
ax.plot(uniform_time, abp2_sg, color="#921b66db", lw=3, alpha=0.7, label="S-G smoothed pressure")
ax.plot(uniform_time, fitted_model, linestyle="--", color="#009F82", lw=2,
        label=fr"First-order fit ($R^2$ = {r2:.3f})")
ax.set_xlabel("Elapsed Time (h)", fontsize=20)
ax.set_ylabel("Pressure (kPa)", fontsize=20)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(fontsize=12, loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "4_FirstOrder_Fit_SG.png"), dpi=dpi)

# === Plot 5: Oxygen Release (µmol) ===
volume_solution_L = 0.006  # 6 mL
concentration_mM = 1.6
V_headspace_L = 0.004  # 4 mL
R = 0.082057  # L·atm/mol·K

max_o2_umol = concentration_mM * volume_solution_L * 1e3
pressure_atm = abp2_sg / 101.325
pressure_atm_dwt = abp2_dwt / 101.325
temperature_K = temperature_interp + 273.15
o2_micromol = ((pressure_atm * V_headspace_L) / (R * temperature_K)) * 1e6
o2_micromol_dwt = ((pressure_atm_dwt * V_headspace_L) / (R * temperature_K)) * 1e6
ratio = volume_solution_L / V_headspace_L

def o2_model(t, N_max, k):
    return N_max * (1 - np.exp(-k * t))

#fit for micromol dwt along with this as well


popt_o2, _ = curve_fit(o2_model, uniform_time, o2_micromol, p0=[np.max(o2_micromol), 0.02])
fitted_o2 = o2_model(uniform_time, *popt_o2)
r2_o2 = r2_score(o2_micromol, fitted_o2)

popt_o2_dwt, _ = curve_fit(o2_model, uniform_time, o2_micromol_dwt, p0=[np.max(o2_micromol_dwt), 0.02])
fitted_o2_dwt = o2_model(uniform_time, *popt_o2_dwt)
r2_o2_dwt = r2_score(o2_micromol_dwt, fitted_o2_dwt)

df_o2_export = pd.DataFrame({
    "Sample": "1.6 mM @ 37°C",
    "Time (h)": uniform_time,
    "SG smoothed pressure (kPa)": abp2_sg,
    "DWT denoised pressure (kPa)": abp2_dwt,
    "Temperature (°C)": temperature_interp,
    "O2 Released (µmol) with SG": o2_micromol,
    "O2 Released (µmol) with DWT": o2_micromol_dwt,
    "% Conversion": (o2_micromol / max_o2_umol) * 100,
    "Soln/Headspace Ratio": ratio,
    "Max O2 Possible (µmol)": max_o2_umol
})
df_o2_export.to_csv("M4_1.6mM_37deg_O2_release_selfcrop.csv", index=False)  #the file is named selfcrop because I cropped the data to remove sensor drift earliern but this will be the official file with everthing

fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)
ax.plot(uniform_time, o2_micromol, label=r"$\mathrm{O_2\ released\ (\mu mol)}$", lw=2.5, color="#1f77b4")
ax.plot(uniform_time, fitted_o2, '--', lw=2, color="#ff7f0e",
        label=fr"First-order fit ($R^2$ = {r2_o2:.3f})")
ax.set_xlabel("Elapsed Time (h)", fontsize=18)
ax.set_ylabel(r"$\mathrm{O_2\ released\ (\mu mol)}$", fontsize=18)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=13, frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "5_O2_release_from_pressure.png"), dpi=dpi)


# === Plot 6: Oxygen Release (µmol) using SG-smoothed pressure only ===
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)

# Plot O₂ released from SG-smoothed signal
ax.plot(uniform_time, o2_micromol, lw=2.5, color="#1f77b4", label=r"$\mathrm{O_2\ released\ (SG)}$")
ax.plot(uniform_time, fitted_o2, '--', lw=2, color="#ff7f0e",
        label=fr"First-order fit (SG, $R^2$ = {r2_o2:.3f})")

ax.set_xlabel("Elapsed Time (h)", fontsize=18)
ax.set_ylabel(r"$\mathrm{O_2\ released\ (\mu mol)}$", fontsize=18)
ax.set_ylim(bottom=0)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=13, frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "6_O2_release_SG_only.png"), dpi=dpi)
plt.show()


# === Plot 7: Compare O₂ Released (SG vs DWT) ===
fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)

ax.plot(uniform_time, o2_micromol, lw=2.5, color="#1f77b4", label=r"$\mathrm{O_2\ (SG)}$")
ax.plot(uniform_time, o2_micromol_dwt, lw=2.5, color="#009E73", label=r"$\mathrm{O_2\ (DWT)}$")
# ax.plot(uniform_time, fitted_o2, '--', lw=1.8, color="#ff7f0e", label=fr"Fit (SG, $R^2$={r2_o2:.3f})")
# ax.plot(uniform_time, fitted_o2_dwt, '--', lw=1.8, color="#d62728", label=fr"Fit (DWT, $R^2$={r2_o2_dwt:.3f})")

ax.set_xlabel("Elapsed Time (h)", fontsize=18)
ax.set_ylabel(r"$\mathrm{O_2\ released\ (\mu mol)}$", fontsize=18)
ax.set_ylim(bottom=0)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=11, frameon=False, loc='lower right', ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "7_O2_release_SG_vs_DWT.png"), dpi=dpi)
plt.show()



# # === File Paths ===
# data_path =  r"C:\Users\chand\Documents\GitHub\Thesis\M3-1.2mM\Pressure\37 C\Data files\overlap_error_removed.csv"
# plot_dir = "processedplots"
# os.makedirs(plot_dir, exist_ok=True)

# # === Load Data ===
# df = pd.read_csv(data_path, parse_dates=["Datetime"])

# # # === Filter: upto certain amount of time if data is large===
# # df = df[df["Elapsed Time (hours)"] <= 90]

# # === Extract Columns ===
# time_raw = df["Elapsed Time (hours)"].values
# abp2_raw = df["ABP2-Sch1 (kPa)"].values
# mpr_raw = df["MPR-Sch2 (kPa)"].values
# temperature = df["Temperature (C)"].values
# humidity = df["Humidity (%RH)"].values

# # === Interpolation to Uniform Time Grid ===
# uniform_time = np.linspace(time_raw.min(), time_raw.max(), int((time_raw.max() - time_raw.min()) * 3600))
# abp2_interp = np.interp(uniform_time, time_raw, abp2_raw)
# temperature_interp = np.interp(uniform_time, time_raw, temperature)

# # === Simple baseline correction (remove temp-based stabilization) ===
# # Use the first N samples as baseline
# baseline_window_size = 180  # adjust as needed (≈3 minutes if 1 Hz sampling)
# baseline_pressure = np.mean(abp2_interp[:baseline_window_size])
# abp2_corrected = abp2_interp - baseline_pressure

# print(f"[Baseline] Using first {baseline_window_size} samples as baseline")
# print(f"[Baseline] Baseline pressure offset: {baseline_pressure:.4f} kPa")

# # === Adaptive SG Window Selection ===
# def compute_snr(signal, noise_region):
#     noise_rms = np.sqrt(np.mean(noise_region**2))
#     signal_p2p = np.ptp(signal)
#     return 20 * np.log10(signal_p2p / noise_rms) if noise_rms != 0 else np.inf

# def evaluate_sg(signal, min_win=13, max_win=99, step=2, polyorder=3):
#     noise_region = signal[-36000:]
#     results = []
#     for w in range(min_win, max_win+1, step):
#         if w >= len(signal): break
#         smoothed = savgol_filter(signal, w, polyorder)
#         snr = compute_snr(smoothed, noise_region)
#         rmse = np.sqrt(mean_squared_error(signal, smoothed))
#         results.append((w, snr, rmse))
#     df = pd.DataFrame(results, columns=["window", "snr", "rmse"])
#     df["snr_norm"] = (df["snr"] - df["snr"].min()) / (df["snr"].max() - df["snr"].min())
#     df["rmse_norm"] = (df["rmse"].max() - df["rmse"]) / (df["rmse"].max() - df["rmse"].min())
#     df["score"] = 0.6 * df["snr_norm"] + 0.4 * df["rmse_norm"]
#     best_win = int(df.loc[df["score"].idxmax()]["window"])
#     best_window_info = df.loc[df['score'].idxmax()]
#     return best_win, best_window_info

# abp2_opt_win, best_window_info = evaluate_sg(abp2_corrected)
# abp2_sg = savgol_filter(abp2_corrected, abp2_opt_win, 3)

# print(f"\nOptimal SG Window Size: {abp2_opt_win}\n")
# print(best_window_info[['snr', 'rmse', 'score']])

# # === DWT Denoising ===
# def dwt_denoise(signal, keep_levels=(3, 4, 5), wavelet="db4"):
#     coeffs = pywt.wavedec(signal, wavelet, level=pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len))
#     for i in range(1, len(coeffs)):
#         if i not in keep_levels:
#             coeffs[i] = np.zeros_like(coeffs[i])
#     return pywt.waverec(coeffs, wavelet)[:len(signal)]

# abp2_dwt = dwt_denoise(abp2_sg)

# # === Save Cleaned Data ===
# clean_df = pd.DataFrame({
#     "Elapsed Time (h)": uniform_time,
#     "ABP2 Raw (baseline-corrected, kPa)": abp2_corrected,
#     "ABP2 SG Smoothed (kPa)": abp2_sg,
#     "ABP2 SG + DWT Denoised (kPa)": abp2_dwt,
#     "Temperature (°C)": temperature_interp
# })

# # === Plotting ===
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman']
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# dpi = 900

# # === Plot 1: Full signal ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)
# ax1.plot(uniform_time, abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
# ax1.plot(uniform_time, abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)
# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.minorticks_on()

# ax2 = ax1.twinx()
# ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1, alpha=0.6, label="Temperature")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.grid(False)

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)

# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "1_M3_37_ABP2_noTempStabilization.png"), dpi=dpi)


# # === 2: plot: Starting from Baseline (T > 36 °C Stabilized)
# # === Crop data from baseline_time_start onward ===
# mask_after_baseline = uniform_time >= baseline_time_start
# plot_time = uniform_time[mask_after_baseline]
# plot_abp2_interp = abp2_interp[mask_after_baseline]
# plot_abp2_corrected = abp2_corrected[mask_after_baseline]
# plot_temperature = smoothed_temp[mask_after_baseline]

# plot_abp2_sg = abp2_sg[mask_after_baseline]
# plot_abp2_dwt = abp2_dwt[mask_after_baseline]

# # === Plot ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# # --- Pressure on left y-axis ---
# # ax1.plot(plot_time, plot_abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
# ax1.plot(plot_time, plot_abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)

# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.set_ylim(np.min(plot_abp2_corrected) - 0.5, np.max(plot_abp2_corrected) + 1.5)
# ax1.yaxis.set_major_locator(MultipleLocator(1.0))
# ax1.grid(axis='y', linestyle='--', alpha=0.6)


# ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax1.minorticks_on()

# # --- Temperature on right y-axis ---
# ax2 = ax1.twinx()
# ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1, alpha=0.6, label="Temperature")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
# ax2.grid(False)
# ax2.tick_params(axis='y', labelcolor="black")
# ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax2.minorticks_on()

# # --- Combine legend only once ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)

# plt.tight_layout()

# # --- Inset: First 5 hours after baseline ---
# inset_mask = plot_time <= (plot_time[0] + 5)
# inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
#                       bbox_to_anchor=(-0.15, 0.2, 1, 1),
#                       bbox_transform=ax1.transAxes, borderpad=0)

# inset_ax.plot(plot_time[inset_mask], plot_abp2_corrected[inset_mask],
#               color="#921b66db", lw=2.5, label="ABP2 Corrected")
# inset_ax.set_ylim(np.min(plot_abp2_corrected[inset_mask]) - 0.1, np.max(plot_abp2_corrected[inset_mask]) + 0.5)
# inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
# inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax.xaxis.set_major_locator(MultipleLocator(1))
# inset_ax.tick_params(labelsize=12)

# inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
# inset_ax.set_title("First 3 h after baseline", fontsize=10)

# inset_ax2 = inset_ax.twinx()
# inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
#                color="tab:red", lw=1.5, alpha=0.6)
# inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
# inset_ax2.set_ylabel("Temp (°C)", fontsize=9)
# inset_ax2.grid(False)
# inset_ax2.tick_params(labelsize=8, labelcolor="black")

# # === Save cropped plot ===
# plt.savefig(os.path.join(plot_dir, "2_M3_37_ABP2_Cropped_from_Baseline.png"), dpi=dpi)

# # === Plot 3: SG-Smoothed Pressure after Baseline ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# # --- Pressure on left y-axis ---
# ax1.plot(plot_time, plot_abp2_sg, label="SG smoothed pressure signal", color="#921b66db", lw=2.5)

# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.set_ylim(np.min(plot_abp2_sg) - 0.5, np.max(plot_abp2_sg) + 1.5)
# ax1.yaxis.set_major_locator(MultipleLocator(1.0))
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
# ax1.tick_params(axis='y', labelcolor="black", labelsize=12)
# ax1.tick_params(axis='x', labelsize=12)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax1.minorticks_on()

# # --- Temperature on right y-axis ---
# ax2 = ax1.twinx()
# ax2.plot(plot_time, plot_temperature, color="tab:red", lw=2, alpha=0.6, label="Temperature profile")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# ax2.grid(False)
# ax2.tick_params(axis='y', labelcolor="black")
# ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax2.minorticks_on()

# # --- Legend ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)

# plt.tight_layout()

# # --- Inset: First 5 hours ---
# inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
#                       bbox_to_anchor=(-0.18, 0.2, 1, 1),
#                       bbox_transform=ax1.transAxes, borderpad=0)

# inset_ax.plot(plot_time[inset_mask], plot_abp2_sg[inset_mask],
#               color="#921b66db", lw=2.5, label="ABP2 SG Smoothed")
# inset_ax.set_ylim(np.min(plot_abp2_sg[inset_mask]) - 0.1, np.max(plot_abp2_sg[inset_mask]) + 0.5)
# inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
# inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax.xaxis.set_major_locator(MultipleLocator(1))
# inset_ax.tick_params(labelsize=12)
# inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
# inset_ax.set_title("First 5 h", fontsize=12)

# inset_ax2 = inset_ax.twinx()
# inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
#                color="tab:red", lw=2, alpha=0.6)
# inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# inset_ax2.set_ylabel("Temp (°C)", fontsize=12)
# inset_ax2.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax2.grid(False)
# inset_ax2.tick_params(labelsize=9, labelcolor="black")

# plt.savefig(os.path.join(plot_dir, "3_M3_37_ABP2_SG_Smoothed.png"), dpi=dpi)

# # === Plot 4: SG + DWT Denoised Pressure after Baseline ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# # --- Pressure on left y-axis ---
# ax1.plot(plot_time, plot_abp2_dwt, label="ABP2 SG + DWT", color="#009E73", lw=2.5)

# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.set_ylim(np.min(plot_abp2_dwt) - 0.5, np.max(plot_abp2_dwt) + 1.5)
# ax1.yaxis.set_major_locator(MultipleLocator(1.0))
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
# ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax1.minorticks_on()

# # --- Temperature on right y-axis ---
# ax2 = ax1.twinx()
# ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1, alpha=0.6, label="Temperature")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# ax2.grid(False)
# ax2.tick_params(axis='y', labelcolor="black")
# ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax2.minorticks_on()

# # --- Legend ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)

# plt.tight_layout()

# # --- Inset: First 5 hours ---
# inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
#                       bbox_to_anchor=(-0.15, 0.2, 1, 1),
#                       bbox_transform=ax1.transAxes, borderpad=0)

# inset_ax.plot(plot_time[inset_mask], plot_abp2_dwt[inset_mask],
#               color="#009E73", lw=2.5, label="ABP2 SG + DWT")
# inset_ax.set_ylim(np.min(plot_abp2_dwt[inset_mask]) - 0.1, np.max(plot_abp2_dwt[inset_mask]) + 0.5)
# inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
# inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax.xaxis.set_major_locator(MultipleLocator(1))
# inset_ax.tick_params(labelsize=12)
# inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
# inset_ax.set_title("First 3 h after baseline", fontsize=10)

# inset_ax2 = inset_ax.twinx()
# inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
#                color="tab:red", lw=1.5, alpha=0.6)
# inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# inset_ax2.set_ylabel("Temp (°C)", fontsize=9)
# inset_ax2.grid(False)
# inset_ax2.tick_params(labelsize=8, labelcolor="black")

# plt.savefig(os.path.join(plot_dir, "4_M3_37_ABP2_SG_DWT.png"), dpi=dpi)


# # === Plot 5: First-order fit to SG-smoothed pressure ===

# # --- First-order kinetic model ---
# def first_order_model(t, Pmax, k):
#     return Pmax * (1 - np.exp(-k * t))

# # Fit to SG-smoothed pressure
# popt, _ = curve_fit(first_order_model, plot_time, plot_abp2_sg, p0=[np.max(plot_abp2_sg), 0.1])
# fitted_model = first_order_model(plot_time, *popt)

# # Metrics
# r2 = r2_score(plot_abp2_sg, fitted_model)
# rmse = np.sqrt(mean_squared_error(plot_abp2_sg, fitted_model))
# n = len(plot_time)
# k_params = len(popt)
# residuals = plot_abp2_sg - fitted_model
# sse = np.sum(residuals**2)
# aic = n * np.log(sse / n) + 2 * k_params

# print(f"[First-Order Fit] R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")

# # Plot
# fig, ax = plt.subplots(figsize=(5, 4.5), dpi=dpi)
# ax.plot(plot_time, plot_abp2_sg, color="#921b66db", lw=3, alpha=0.7, label=fr"S-G smoothed pressure signal" + "\n")
# ax.plot(plot_time, fitted_model, linestyle="--", color="#009F82", lw=2,
#         label=fr"First-order fit ($R^2$ = {r2:.3f})" + "\n" + r"$P(t) = P_{\max}(1 - e^{-kt})$" + "\n")

# ax.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax.set_ylabel("Pressure (kPa)", fontsize=20)
# ax.set_ylim(np.min(plot_abp2_sg) - 0.5, np.max(plot_abp2_sg) + 0.5)
# ax.yaxis.set_major_locator(MultipleLocator(1.0))
# ax.grid(axis='y', linestyle='--', alpha=0.6)
# ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax.minorticks_on()
# ax.legend(fontsize=14, loc="lower right", frameon=False)

# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "5_M3_37_FirstOrder_Fit_SG.png"), dpi=dpi)



# # === Plot 6: Oxygen Release (µmol) from Pressure ===

# # Constants
# volume_solution_L = 0.006 # 6ml
# concentration_mM = 1.2

# max_o2_umol = concentration_mM * volume_solution_L * 1e3 # micromol

# # pressure to micromol using ideal gas law
# V_headspace_L = 0.004  # 4 ml = 0.004 L
# R = 0.082057  # L·atm/mol·K

# pressure_atm = plot_abp2_sg / 101.325      # Convert pressure (kPa) to atm 
# temperature_K = plot_temperature + 273.15   # temperature to Kelvin

# # Compute µmol O₂
# o2_micromol = ((pressure_atm * V_headspace_L) / (R * temperature_K)) * 1e6

# ratio = volume_solution_L / V_headspace_L

# # Fit O₂ release to first-order model
# def o2_model(t, N_max, k):
#     return N_max * (1 - np.exp(-k * t))

# popt_o2, _ = curve_fit(o2_model, plot_time, o2_micromol, p0=[np.max(o2_micromol), 0.02])
# fitted_o2 = o2_model(plot_time, *popt_o2)
# r2_o2 = r2_score(o2_micromol, fitted_o2)


# df_o2_export = pd.DataFrame({
#     "Sample": "1.2 mM @ 37deg",
#     "Time (h)": plot_time,
#     "DWT denoised pressure (kPa)": plot_abp2_dwt,
#     "SG smoothed pressure (kPa)": plot_abp2_sg,
#     "calibrated temperature (C)": plot_temperature,
#     "O2 Released (µmol)-SG": o2_micromol,
#     "% Conversion": (o2_micromol / max_o2_umol) * 100,
#     "Soln/Headspace Ratio": ratio,
#     "Max O2 Possible (µmol)": max_o2_umol
# })

# df_o2_export.to_csv("M3_1.2mM_37deg_pressure_o2_release.csv", index=False)


# # Plot
# fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)
# ax.plot(plot_time, o2_micromol, label="O2 released (µmol)", lw=2.5, color="#1f77b4")
# ax.plot(plot_time, fitted_o2, '--', lw=2, color="#ff7f0e", label=fr"First-order fit ($R^2$ = {r2_o2:.3f})")
# # ax.axhline(o2_theoretical, linestyle='--', color='gray', linewidth=1.5,
# #            label=f"Theoretical max: {o2_theoretical:.2f} µmol")

# ax.set_xlabel("Elapsed Time (h)", fontsize=18)
# ax.set_ylabel("O2 released (µmol)", fontsize=18)
# ax.set_ylim(bottom=0)
# ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax.minorticks_on()
# ax.grid(axis='y', linestyle='--', alpha=0.5)
# ax.legend(fontsize=13, frameon=False, loc='lower right')

# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "6_M3_37_O2_release_from_pressure.png"), dpi=dpi)

























# # ==========================================
# # PRESSURE ANALYSIS SCRIPT WITH SG somoothing and DWT
# # ==========================================
# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# from scipy.optimize import curve_fit, minimize
# from sklearn.metrics import mean_squared_error, r2_score
# import pywt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from scipy.ndimage import uniform_filter1d
# from scipy.ndimage import label
# from matplotlib import rcParams
# from matplotlib.ticker import MultipleLocator

# # === File Paths ===
# data_path =  r"C:\Users\chand\Documents\GitHub\Thesis\M3-1.2mM\Sample_M3\M3_1.2mM\Pressure_37deg\raw_meas\merged_pressure_log.csv"
# plot_dir = "processedplots"
# os.makedirs(plot_dir, exist_ok=True)

# # === Load Data ===
# df = pd.read_csv(data_path, parse_dates=["Datetime"])

# # === Filter: upto certain amount of time if data is large===
# df = df[df["Elapsed Time (hours)"] <= 90]

# # === Extract Columns ===
# time_raw = df["Elapsed Time (hours)"].values
# abp2_raw = df["ABP2-Sch1 (kPa)"].values
# mpr_raw = df["MPR-Sch2 (kPa)"].values
# temperature = df["Temperature (C)"].values
# humidity = df["Humidity (%RH)"].values

# # === Interpolation to Uniform Time Grid ===
# uniform_time = np.linspace(time_raw.min(), time_raw.max(), int((time_raw.max() - time_raw.min()) * 3600))
# abp2_interp = np.interp(uniform_time, time_raw, abp2_raw)
# temperature_interp = np.interp(uniform_time, time_raw, temperature)

# # Smooth temperature to reduce noise (5 sec window)
# smoothed_temp = uniform_filter1d(temperature_interp, size=300)

# # === baseline: Auto Baseline Correction Based on Temp Stabilization ===
# # === 1. Identify longest stable run with T > 36°C ===
# binary_mask = smoothed_temp > 36.0
# labeled, n_features = label(binary_mask)

# if n_features == 0:
#     raise ValueError("No region with temperature > 36°C found.")

# lengths = [(i, np.sum(labeled == i)) for i in range(1, n_features + 1)]
# longest_stable = max(lengths, key=lambda x: x[1])[0]
# stable_idx = np.where(labeled == longest_stable)[0]

# # === 2. Start 5 minutes after stable region begins ===
# sampling_rate_hz = len(uniform_time) / (uniform_time[-1] - uniform_time[0])     # samples per hour
# offset_samples = int(5 * 60 * sampling_rate_hz / 3600)                          # 5 minutes offset in samples

# start_idx = stable_idx[0] + offset_samples

# # === 3. Check bounds ===
# if start_idx >= len(uniform_time):
#     raise IndexError("Start index after temperature stabilization exceeds data range.")

# baseline_window_size = 180  # samples (adjust if needed)
# end_idx = min(start_idx + baseline_window_size, len(uniform_time))

# if end_idx - start_idx < 60:
#     raise ValueError("Baseline window too short after bounds check.")

# # === 4. Compute baseline and correct ===
# baseline_time_start = uniform_time[start_idx]
# baseline_time_end = uniform_time[end_idx - 1]
# baseline_pressure = np.mean(abp2_interp[start_idx:end_idx])
# abp2_corrected = abp2_interp - baseline_pressure

# # === 5. Print diagnostic info ===
# print(f"[Auto-Baseline] T > 36°C stabilized at t ≈ {uniform_time[stable_idx[0]]:.3f} h")
# print(f"[Auto-Baseline] Using baseline window: {baseline_time_start:.3f} – {baseline_time_end:.3f} h")
# print(f"[Auto-Baseline] Baseline pressure offset: {baseline_pressure:.4f} kPa")
# print(f"Pressure at baseline window start: {abp2_corrected[start_idx]:.4f} kPa")

# # === Adaptive SG Window Selection ===
# def compute_snr(signal, noise_region):
#     noise_rms = np.sqrt(np.mean(noise_region**2))
#     signal_p2p = np.ptp(signal)
#     return 20 * np.log10(signal_p2p / noise_rms) if noise_rms != 0 else np.inf

# def evaluate_sg(signal, min_win=13, max_win=99, step=2, polyorder=3):
#     noise_region = signal[-36000:]
#     results = []
#     for w in range(min_win, max_win+1, step):
#         if w >= len(signal): break
#         smoothed = savgol_filter(signal, w, polyorder)
#         snr = compute_snr(smoothed, noise_region)
#         rmse = np.sqrt(mean_squared_error(signal, smoothed))
#         results.append((w, snr, rmse))
#     df = pd.DataFrame(results, columns=["window", "snr", "rmse"])
#     df["snr_norm"] = (df["snr"] - df["snr"].min()) / (df["snr"].max() - df["snr"].min())
#     df["rmse_norm"] = (df["rmse"].max() - df["rmse"]) / (df["rmse"].max() - df["rmse"].min())
#     df["score"] = 0.6 * df["snr_norm"] + 0.4 * df["rmse_norm"]
#     best_win = int(df.loc[df["score"].idxmax()]["window"])
#     best_window_info = df.loc[df['score'].idxmax()]
#     return best_win, best_window_info

# abp2_opt_win, best_window_info = evaluate_sg(abp2_corrected)
# abp2_sg = savgol_filter(abp2_corrected, abp2_opt_win, 3)

# print(f"\nOptimal SG Window Size: {abp2_opt_win}\n")
# print(best_window_info[['snr', 'rmse', 'score']])

# # === DWT Denoising ===
# def dwt_denoise(signal, keep_levels=(3, 4, 5), wavelet="db4"):
#     coeffs = pywt.wavedec(signal, wavelet, level=pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len))
#     for i in range(1, len(coeffs)):
#         if i not in keep_levels:
#             coeffs[i] = np.zeros_like(coeffs[i])
#     return pywt.waverec(coeffs, wavelet)[:len(signal)]

# abp2_dwt = dwt_denoise(abp2_sg)

# # === Save Cleaned Data ===
# # Export baseline-corrected, smoothed signal, to save the cleaned version for modeling:
# clean_df = pd.DataFrame({
#     "Elapsed Time (h)": uniform_time,
#     "ABP2 Raw (baseline-corrected, kPa)": abp2_corrected,
#     "ABP2 SG Smoothed (kPa)": abp2_sg,
#     "ABP2 SG + DWT Denoised (kPa)": abp2_dwt,
#     "Temperature (°C)": temperature_interp
# })

# # === Plotting ===
# # aesthetics & fonts
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman']
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# dpi = 900

# # === 1: Plot ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# # --- Pressure on left y-axis ---
# ax1.plot(uniform_time, abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
# ax1.plot(uniform_time, abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)

# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.set_ylim(np.min(abp2_corrected), np.max(abp2_interp)+1.5 )
# ax1.yaxis.set_major_locator(MultipleLocator(1.0))
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
# ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax1.minorticks_on()

# # --- Temperature on right y-axis ---
# ax2 = ax1.twinx()
# ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1, alpha=0.6, label="Temperature")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.grid(False)
# ax2.tick_params(axis='y', labelcolor="black")
# ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax2.minorticks_on()

# # --- Combine legend ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)

# # plt.title("ABP2 Thermal Correction with Temperature Overlay", fontsize=13)
# plt.tight_layout()

# # --- Inset: First 15 minutes ---
# inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
#                       bbox_to_anchor=(-0.12, 0.2, 1, 1),
#                       bbox_transform=ax1.transAxes, borderpad=0)

# inset_mask = (uniform_time <= 5)
# inset_ax.plot(uniform_time[inset_mask], abp2_corrected[inset_mask],
#               color="#921b66db", lw=1.5, label="ABP2 Corrected")
# inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
# inset_ax.xaxis.set_major_locator(MultipleLocator(1))
# inset_ax.tick_params(labelsize=12)
# inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
# inset_ax.set_title("First 5 hours", fontsize=10)

# # --- Temperature in inset (right y-axis) ---
# inset_ax2 = inset_ax.twinx()
# inset_ax2.plot(uniform_time[inset_mask], temperature_interp[inset_mask],
#                color="tab:red", lw=1.2, alpha=0.6)
# inset_ax2.set_ylabel("Temp (°C)", fontsize=8)
# inset_ax2.grid(False)
# inset_ax2.tick_params(labelsize=8, labelcolor="black")

# plt.savefig(os.path.join(plot_dir, "1_M3_37_ABP2_thermal_overlay_with_inset.png"), dpi=dpi)

# # === 2: plot: Starting from Baseline (T > 36 °C Stabilized)
# # === Crop data from baseline_time_start onward ===
# mask_after_baseline = uniform_time >= baseline_time_start
# plot_time = uniform_time[mask_after_baseline]
# plot_abp2_interp = abp2_interp[mask_after_baseline]
# plot_abp2_corrected = abp2_corrected[mask_after_baseline]
# plot_temperature = smoothed_temp[mask_after_baseline]

# plot_abp2_sg = abp2_sg[mask_after_baseline]
# plot_abp2_dwt = abp2_dwt[mask_after_baseline]

# # === Plot ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# # --- Pressure on left y-axis ---
# # ax1.plot(plot_time, plot_abp2_interp, label="ABP2 Raw", color="#1f78b4", lw=2.5)
# ax1.plot(plot_time, plot_abp2_corrected, label="ABP2 Corrected", color="#921b66db", lw=2.5)

# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.set_ylim(np.min(plot_abp2_corrected) - 0.5, np.max(plot_abp2_corrected) + 1.5)
# ax1.yaxis.set_major_locator(MultipleLocator(1.0))
# ax1.grid(axis='y', linestyle='--', alpha=0.6)


# ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax1.minorticks_on()

# # --- Temperature on right y-axis ---
# ax2 = ax1.twinx()
# ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1, alpha=0.6, label="Temperature")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
# ax2.grid(False)
# ax2.tick_params(axis='y', labelcolor="black")
# ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax2.minorticks_on()

# # --- Combine legend only once ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)

# plt.tight_layout()

# # --- Inset: First 5 hours after baseline ---
# inset_mask = plot_time <= (plot_time[0] + 5)
# inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
#                       bbox_to_anchor=(-0.15, 0.2, 1, 1),
#                       bbox_transform=ax1.transAxes, borderpad=0)

# inset_ax.plot(plot_time[inset_mask], plot_abp2_corrected[inset_mask],
#               color="#921b66db", lw=2.5, label="ABP2 Corrected")
# inset_ax.set_ylim(np.min(plot_abp2_corrected[inset_mask]) - 0.1, np.max(plot_abp2_corrected[inset_mask]) + 0.5)
# inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
# inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax.xaxis.set_major_locator(MultipleLocator(1))
# inset_ax.tick_params(labelsize=12)

# inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
# inset_ax.set_title("First 3 h after baseline", fontsize=10)

# inset_ax2 = inset_ax.twinx()
# inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
#                color="tab:red", lw=1.5, alpha=0.6)
# inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature)+0.2)
# inset_ax2.set_ylabel("Temp (°C)", fontsize=9)
# inset_ax2.grid(False)
# inset_ax2.tick_params(labelsize=8, labelcolor="black")

# # === Save cropped plot ===
# plt.savefig(os.path.join(plot_dir, "2_M3_37_ABP2_Cropped_from_Baseline.png"), dpi=dpi)

# # === Plot 3: SG-Smoothed Pressure after Baseline ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# # --- Pressure on left y-axis ---
# ax1.plot(plot_time, plot_abp2_sg, label="SG smoothed pressure signal", color="#921b66db", lw=2.5)

# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.set_ylim(np.min(plot_abp2_sg) - 0.5, np.max(plot_abp2_sg) + 1.5)
# ax1.yaxis.set_major_locator(MultipleLocator(1.0))
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
# ax1.tick_params(axis='y', labelcolor="black", labelsize=12)
# ax1.tick_params(axis='x', labelsize=12)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax1.minorticks_on()

# # --- Temperature on right y-axis ---
# ax2 = ax1.twinx()
# ax2.plot(plot_time, plot_temperature, color="tab:red", lw=2, alpha=0.6, label="Temperature profile")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# ax2.grid(False)
# ax2.tick_params(axis='y', labelcolor="black")
# ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax2.minorticks_on()

# # --- Legend ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="lower right", frameon=False, ncol=2)

# plt.tight_layout()

# # --- Inset: First 5 hours ---
# inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
#                       bbox_to_anchor=(-0.18, 0.2, 1, 1),
#                       bbox_transform=ax1.transAxes, borderpad=0)

# inset_ax.plot(plot_time[inset_mask], plot_abp2_sg[inset_mask],
#               color="#921b66db", lw=2.5, label="ABP2 SG Smoothed")
# inset_ax.set_ylim(np.min(plot_abp2_sg[inset_mask]) - 0.1, np.max(plot_abp2_sg[inset_mask]) + 0.5)
# inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
# inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax.xaxis.set_major_locator(MultipleLocator(1))
# inset_ax.tick_params(labelsize=12)
# inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
# inset_ax.set_title("First 5 h", fontsize=12)

# inset_ax2 = inset_ax.twinx()
# inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
#                color="tab:red", lw=2, alpha=0.6)
# inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# inset_ax2.set_ylabel("Temp (°C)", fontsize=12)
# inset_ax2.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax2.grid(False)
# inset_ax2.tick_params(labelsize=9, labelcolor="black")

# plt.savefig(os.path.join(plot_dir, "3_M3_37_ABP2_SG_Smoothed.png"), dpi=dpi)

# # === Plot 4: SG + DWT Denoised Pressure after Baseline ===
# fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)

# # --- Pressure on left y-axis ---
# ax1.plot(plot_time, plot_abp2_dwt, label="ABP2 SG + DWT", color="#009E73", lw=2.5)

# ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
# ax1.set_ylim(np.min(plot_abp2_dwt) - 0.5, np.max(plot_abp2_dwt) + 1.5)
# ax1.yaxis.set_major_locator(MultipleLocator(1.0))
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
# ax1.tick_params(axis='y', labelcolor="black", labelsize=10)
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax1.minorticks_on()

# # --- Temperature on right y-axis ---
# ax2 = ax1.twinx()
# ax2.plot(plot_time, plot_temperature, color="tab:red", lw=1, alpha=0.6, label="Temperature")
# ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
# ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# ax2.grid(False)
# ax2.tick_params(axis='y', labelcolor="black")
# ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax2.minorticks_on()

# # --- Legend ---
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)

# plt.tight_layout()

# # --- Inset: First 5 hours ---
# inset_ax = inset_axes(ax1, width="35%", height="38%", loc="lower right",
#                       bbox_to_anchor=(-0.15, 0.2, 1, 1),
#                       bbox_transform=ax1.transAxes, borderpad=0)

# inset_ax.plot(plot_time[inset_mask], plot_abp2_dwt[inset_mask],
#               color="#009E73", lw=2.5, label="ABP2 SG + DWT")
# inset_ax.set_ylim(np.min(plot_abp2_dwt[inset_mask]) - 0.1, np.max(plot_abp2_dwt[inset_mask]) + 0.5)
# inset_ax.set_ylabel("Pressure (kPa)", fontsize=12)
# inset_ax.yaxis.set_major_locator(MultipleLocator(0.5))
# inset_ax.xaxis.set_major_locator(MultipleLocator(1))
# inset_ax.tick_params(labelsize=12)
# inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
# inset_ax.set_title("First 3 h after baseline", fontsize=10)

# inset_ax2 = inset_ax.twinx()
# inset_ax2.plot(plot_time[inset_mask], plot_temperature[inset_mask],
#                color="tab:red", lw=1.5, alpha=0.6)
# inset_ax2.set_ylim(np.min(plot_temperature) - 3, np.max(plot_temperature) + 0.2)
# inset_ax2.set_ylabel("Temp (°C)", fontsize=9)
# inset_ax2.grid(False)
# inset_ax2.tick_params(labelsize=8, labelcolor="black")

# plt.savefig(os.path.join(plot_dir, "4_M3_37_ABP2_SG_DWT.png"), dpi=dpi)


# # === Plot 5: First-order fit to SG-smoothed pressure ===

# # --- First-order kinetic model ---
# def first_order_model(t, Pmax, k):
#     return Pmax * (1 - np.exp(-k * t))

# # Fit to SG-smoothed pressure
# popt, _ = curve_fit(first_order_model, plot_time, plot_abp2_sg, p0=[np.max(plot_abp2_sg), 0.1])
# fitted_model = first_order_model(plot_time, *popt)

# # Metrics
# r2 = r2_score(plot_abp2_sg, fitted_model)
# rmse = np.sqrt(mean_squared_error(plot_abp2_sg, fitted_model))
# n = len(plot_time)
# k_params = len(popt)
# residuals = plot_abp2_sg - fitted_model
# sse = np.sum(residuals**2)
# aic = n * np.log(sse / n) + 2 * k_params

# print(f"[First-Order Fit] R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")

# # Plot
# fig, ax = plt.subplots(figsize=(5, 4.5), dpi=dpi)
# ax.plot(plot_time, plot_abp2_sg, color="#921b66db", lw=3, alpha=0.7, label=fr"S-G smoothed pressure signal" + "\n")
# ax.plot(plot_time, fitted_model, linestyle="--", color="#009F82", lw=2,
#         label=fr"First-order fit ($R^2$ = {r2:.3f})" + "\n" + r"$P(t) = P_{\max}(1 - e^{-kt})$" + "\n")

# ax.set_xlabel("Elapsed Time (h)", fontsize=20)
# ax.set_ylabel("Pressure (kPa)", fontsize=20)
# ax.set_ylim(np.min(plot_abp2_sg) - 0.5, np.max(plot_abp2_sg) + 0.5)
# ax.yaxis.set_major_locator(MultipleLocator(1.0))
# ax.grid(axis='y', linestyle='--', alpha=0.6)
# ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax.minorticks_on()
# ax.legend(fontsize=14, loc="lower right", frameon=False)

# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "5_M3_37_FirstOrder_Fit_SG.png"), dpi=dpi)



# # === Plot 6: Oxygen Release (µmol) from Pressure ===

# # Constants
# volume_solution_L = 0.006 # 6ml
# concentration_mM = 1.2

# max_o2_umol = concentration_mM * volume_solution_L * 1e3 # micromol

# # pressure to micromol using ideal gas law
# V_headspace_L = 0.004  # 4 ml = 0.004 L
# R = 0.082057  # L·atm/mol·K

# pressure_atm = plot_abp2_dwt / 101.325      # Convert pressure (kPa) to atm 
# temperature_K = plot_temperature + 273.15   # temperature to Kelvin

# # Compute µmol O₂
# o2_micromol = ((pressure_atm * V_headspace_L) / (R * temperature_K)) * 1e6

# ratio = volume_solution_L / V_headspace_L

# # Fit O₂ release to first-order model
# def o2_model(t, N_max, k):
#     return N_max * (1 - np.exp(-k * t))

# popt_o2, _ = curve_fit(o2_model, plot_time, o2_micromol, p0=[np.max(o2_micromol), 0.02])
# fitted_o2 = o2_model(plot_time, *popt_o2)
# r2_o2 = r2_score(o2_micromol, fitted_o2)


# df_o2_export = pd.DataFrame({
#     "Sample": "1.2 mM @ 37deg",
#     "Time (h)": plot_time,
#     "DWT denoised pressure (kPa)": plot_abp2_dwt,
#     "calibrated temperature (C)": plot_temperature,
#     "O2 Released (µmol)": o2_micromol,
#     "% Conversion": (o2_micromol / max_o2_umol) * 100,
#     "Soln/Headspace Ratio": ratio,
#     "Max O2 Possible (µmol)": max_o2_umol
# })

# df_o2_export.to_csv("M3_1.2mM_37deg_pressure_o2_release.csv", index=False)


# # Plot
# fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)
# ax.plot(plot_time, o2_micromol, label="O2 released (µmol)", lw=2.5, color="#1f77b4")
# ax.plot(plot_time, fitted_o2, '--', lw=2, color="#ff7f0e", label=fr"First-order fit ($R^2$ = {r2_o2:.3f})")
# # ax.axhline(o2_theoretical, linestyle='--', color='gray', linewidth=1.5,
# #            label=f"Theoretical max: {o2_theoretical:.2f} µmol")

# ax.set_xlabel("Elapsed Time (h)", fontsize=18)
# ax.set_ylabel("O2 released (µmol)", fontsize=18)
# ax.set_ylim(bottom=0)
# ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
# ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
# ax.minorticks_on()
# ax.grid(axis='y', linestyle='--', alpha=0.5)
# ax.legend(fontsize=13, frameon=False, loc='lower right')

# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "6_M3_37_O2_release_from_pressure.png"), dpi=dpi)