# ==========================================
# PRESSURE ANALYSIS SCRIPT (SG tweaks: Option A)
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import pywt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

# === File Paths ===
data_path = r"C:\Users\chand\Documents\GitHub\Thesis\M3-1.2mM\Sample_M3\M3_1.2mM\Pressure_37deg\raw_meas\raw_negative_dropped.csv"
plot_dir  = "processedplots"
os.makedirs(plot_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(data_path, parse_dates=["Datetime"])

# === Filter: up to 90 h (optional) ===
df = df[df["Elapsed Time (hours)"] <= 90]

# === Extract Columns ===
time_raw   = df["Elapsed Time (hours)"].values
abp2_raw   = pd.to_numeric(df["ABP2-Sch1 (kPa)"], errors="coerce").values
temperature = pd.to_numeric(df["Temperature (C)"], errors="coerce").values

# === Interpolate to a uniform time grid (1 Hz equivalent if your data is ~1s; set by range*3600) ===
uniform_time        = np.linspace(time_raw.min(), time_raw.max(), int((time_raw.max() - time_raw.min()) * 3600))
abp2_interp         = np.interp(uniform_time, time_raw, abp2_raw)
temperature_interp  = np.interp(uniform_time, time_raw, temperature)

# ------------------------------------------------------------------
# === Baseline: average of the first 5 minutes ===
baseline_minutes   = 5
baseline_end_time  = uniform_time[0] + baseline_minutes / 60.0  # hours
baseline_mask      = (uniform_time <= baseline_end_time)

if baseline_mask.sum() < 10:
    raise ValueError("Not enough samples in the first 5 minutes for baseline.")

baseline_pressure  = np.mean(abp2_interp[baseline_mask])
abp2_corrected     = abp2_interp - baseline_pressure
baseline_time_start = baseline_end_time

print(f"[Baseline] Using first {baseline_minutes} min "
      f"({uniform_time[0]:.3f}–{baseline_end_time:.3f} h). "
      f"Baseline offset = {baseline_pressure:.4f} kPa")
# ------------------------------------------------------------------

# === Option A: SG tweaks (edge-safe + time-based window selection) ===
def win_sec_to_samples(seconds, fs_hz, poly=3):
    """Convert seconds to a valid odd Savitzky-Golay window length (>= poly+2)."""
    w = int(round(seconds * fs_hz))
    if w % 2 == 0:
        w += 1
    w = max(w, poly + 3 if (poly + 3) % 2 == 1 else poly + 4)  # ensure >= poly+2 and odd
    return w

# sampling rate (samples per second) from uniform grid
total_seconds = (uniform_time[-1] - uniform_time[0]) * 3600.0
fs_hz = len(uniform_time) / total_seconds if total_seconds > 0 else 1.0

sg_poly   = 3
cand_secs = [60, 120, 180, 300]  # candidate windows: 1–5 minutes
cand_win  = [win_sec_to_samples(s, fs_hz, sg_poly) for s in cand_secs]

# use last 30 minutes of corrected signal as a "noise" proxy (adjust if preferred)
noise_len = int(max(1, round(30*60*fs_hz)))
noise_region = abp2_corrected[-noise_len:]

best_score, best_w, abp2_sg = -np.inf, None, None
for w in cand_win:
    sm = savgol_filter(abp2_corrected, w, sg_poly, mode='interp')
    snr = 20*np.log10(np.ptp(sm) / (np.sqrt((noise_region**2).mean()) + 1e-12))
    rmse = np.sqrt(mean_squared_error(abp2_corrected, sm))
    score = 0.6 * snr + 0.4 * (1.0 / (rmse + 1e-12))
    if score > best_score:
        best_score, best_w, abp2_sg = score, w, sm

print(f"[SG] fs≈{fs_hz:.2f} Hz, best window={best_w} samples (~{best_w/fs_hz:.1f}s), poly={sg_poly}")

# === DWT denoising (keep approximation, soften details; then re-align baseline) ===
def dwt_soft_denoise_dc_safe(x, wavelet="db4", mode="periodization", level=None):
    if level is None:
        level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x, wavelet, mode=mode, level=level)
    cA, details = coeffs[0], coeffs[1:]

    sigma = np.median(np.abs(details[-1])) / 0.6745 + 1e-12
    thr   = sigma * np.sqrt(2*np.log(len(x)))  # try scale 0.5–1.5 if needed
    details_th = [pywt.threshold(cD, thr, mode="soft") for cD in details]

    y = pywt.waverec([cA] + details_th, wavelet, mode=mode)
    return y[:len(x)]

abp2_dwt = dwt_soft_denoise_dc_safe(abp2_sg, wavelet="db4", mode="periodization")

# Re-align DWT to same baseline window as SG to avoid DC drift
sg_base  = np.mean(abp2_sg[baseline_mask])
dwt_base = np.mean(abp2_dwt[baseline_mask])
abp2_dwt = abp2_dwt - dwt_base + sg_base

# === Save Cleaned Data (full uniform grid) ===
clean_df = pd.DataFrame({
    "Elapsed Time (h)": uniform_time,
    "ABP2 Raw (baseline-corrected, kPa)": abp2_corrected,
    "ABP2 SG Smoothed (kPa)": abp2_sg,
    "ABP2 SG + DWT Denoised (kPa)": abp2_dwt,
    "Temperature (°C)": temperature_interp
})
clean_df.to_csv(os.path.join(plot_dir, "clean_series.csv"), index=False)

# === Plotting aesthetics ===
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Times New Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
dpi = 900

# === Plot 1: Raw vs Corrected + Temperature overlay ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)
ax1.plot(uniform_time, abp2_interp,     label="ABP2 Raw",       color="#1f78b4", lw=2.2)
ax1.plot(uniform_time, abp2_corrected,  label="ABP2 Corrected", color="#921b66db", lw=2.2)

ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.minorticks_on()

ax2 = ax1.twinx()
ax2.plot(uniform_time, temperature_interp, color="tab:red", lw=1, alpha=0.6, label="Temperature")
ax2.set_ylabel("Temperature (°C)", fontsize=20, color="black")
ax2.grid(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right", frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "1_ABP2_raw_vs_corrected.png"), dpi=dpi)

# === Crop from end of baseline window onward (used for later plots/fits) ===
mask_after_baseline = uniform_time >= baseline_time_start
plot_time         = uniform_time[mask_after_baseline]
plot_temperature  = temperature_interp[mask_after_baseline]
plot_abp2_corrected = abp2_corrected[mask_after_baseline]
plot_abp2_sg        = abp2_sg[mask_after_baseline]
plot_abp2_dwt       = abp2_dwt[mask_after_baseline]

# === Plot 2: SG only ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)
ax1.plot(plot_time, plot_abp2_sg, label="SG smoothed pressure", color="#921b66db", lw=2.5)
ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20, color="black")
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax1.minorticks_on()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "2_ABP2_SG.png"), dpi=dpi)

# === Plot 3: SG + DWT ===
fig, ax1 = plt.subplots(figsize=(7, 5), dpi=dpi)
ax1.plot(plot_time, plot_abp2_sg,  label="SG smoothed", color="#1f77b4", lw=2.2)
ax1.plot(plot_time, plot_abp2_dwt, label="SG + DWT (aligned)", color="#ff7f0e", lw=2.2)
ax1.set_xlabel("Elapsed Time (h)", fontsize=20)
ax1.set_ylabel("Pressure (kPa)", fontsize=20)
ax1.yaxis.set_major_locator(MultipleLocator(1.0))
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.legend(frameon=False, fontsize=12, loc="best")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "3_ABP2_SG_vs_DWT.png"), dpi=dpi)

# === Plot 4: First-order fit to SG-smoothed pressure ===
def first_order_model(t, Pmax, k):
    return Pmax * (1 - np.exp(-k * t))

popt, _      = curve_fit(first_order_model, plot_time, plot_abp2_sg, p0=[np.max(plot_abp2_sg), 0.1])
fitted_model = first_order_model(plot_time, *popt)

r2   = r2_score(plot_abp2_sg, fitted_model)
rmse = np.sqrt(mean_squared_error(plot_abp2_sg, fitted_model))
n    = len(plot_time)
k_params = len(popt)
residuals = plot_abp2_sg - fitted_model
sse  = np.sum(residuals**2)
aic  = n * np.log(sse / n) + 2 * k_params

print(f"[First-Order Fit] R² = {r2:.4f}, RMSE = {rmse:.4f}, AIC = {aic:.2f}")

fig, ax = plt.subplots(figsize=(5, 4.5), dpi=dpi)
ax.plot(plot_time, plot_abp2_sg, color="#921b66db", lw=3, alpha=0.7, label="SG smoothed")
ax.plot(plot_time, fitted_model, linestyle="--", color="#009F82", lw=2,
        label=fr"First-order fit ($R^2$ = {r2:.3f})" + "\n" + r"$P(t) = P_{\max}(1 - e^{-kt})$")
ax.set_xlabel("Elapsed Time (h)", fontsize=20)
ax.set_ylabel("Pressure (kPa)", fontsize=20)
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax.minorticks_on()
ax.legend(fontsize=14, loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "4_FirstOrderFit_SG.png"), dpi=dpi)

# === Plot 5: Oxygen Release (µmol) from Pressure, using DWT-aligned ===
# Constants
volume_solution_L = 0.006  # 6 mL
concentration_mM  = 1.2
max_o2_umol       = concentration_mM * volume_solution_L * 1e3  # µmol

V_headspace_L = 0.004  # 4 mL
R = 0.082057           # L·atm·mol⁻¹·K⁻¹

pressure_atm  = plot_abp2_dwt / 101.325
temperature_K = plot_temperature + 273.15
o2_micromol   = ((pressure_atm * V_headspace_L) / (R * temperature_K)) * 1e6
ratio         = volume_solution_L / V_headspace_L

def o2_model(t, N_max, k):
    return N_max * (1 - np.exp(-k * t))

popt_o2, _ = curve_fit(o2_model, plot_time, o2_micromol, p0=[np.max(o2_micromol), 0.02])
fitted_o2  = o2_model(plot_time, *popt_o2)
r2_o2      = r2_score(o2_micromol, fitted_o2)

# Export O2 table
df_o2_export = pd.DataFrame({
    "Sample": "1.2 mM @ 37deg",
    "Time (h)": plot_time,
    "DWT denoised pressure (kPa)": plot_abp2_dwt,
    "calibrated temperature (C)": plot_temperature,
    "O2 Released (µmol)": o2_micromol,
    "% Conversion": (o2_micromol / max_o2_umol) * 100,
    "Smoothed pressure from just SG (kPa)": plot_abp2_sg,
    "Soln/Headspace Ratio": ratio,
    "Max O2 Possible (µmol)": max_o2_umol
})
df_o2_export.to_csv("M3_1.2mM_37deg_pressure_o2_release.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 5), dpi=dpi)
ax.plot(plot_time, o2_micromol, label="O2 released (µmol)", lw=2.5, color="#1f77b4")
ax.plot(plot_time, fitted_o2, '--', lw=2, color="#ff7f0e", label=fr"First-order fit ($R^2$ = {r2_o2:.3f})")
ax.set_xlabel("Elapsed Time (h)", fontsize=18)
ax.set_ylabel("O2 released (µmol)", fontsize=18)
ax.set_ylim(bottom=0)
ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
ax.minorticks_on()
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=13, frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "5_O2_release_from_pressure.png"), dpi=dpi)
