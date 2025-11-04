# ==========================================
# FULL PRESSURE ANALYSIS SCRIPT (UNIFIED CSV WITH TEMPERATURE)
# ==========================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
from scipy.signal import savgol_filter, find_peaks, correlate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import mean_squared_error, r2_score
import pywt
from datetime import datetime
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.constants import R
from matplotlib.patheffects import withStroke
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import seaborn as sns
# === Legend with turbo gradient line ===
from matplotlib.legend_handler import HandlerLineCollection
import matplotlib.patches as Patch
from matplotlib.legend_handler import HandlerBase

# === Folder Paths ===
# data_dir = "raw_meas"
plot_dir = r"C:\Users\chand\Documents\GitHub\Thesis\M4-1.6 mM\RT\plots"
csv_output_dir = r"C:\Users\chand\Documents\GitHub\Thesis\M4-1.6 mM\RT\denoised csv"

# ***********************************************
'''
Baseline adjustment steps:
 - remove the first 0.5 h of transient (but keeps true absolute time).
 - Look for pressure point zone when the raw started to rise
 - Searche between 5–7 h for the lowest pressure point.
 - Average a ± 0.25 h window around that minimum to suppress noise.
 - Subtract that mean value as the baseline offset.
 - Return the corrected signal, the baseline value, and the time of the baseline center.
'''
# ***********************************************

# === 1. Load Data from Unified CSV (Pressure + Temperature) ===
def load_and_prepare_data():
    sample_df = pd.read_csv(r"C:\Users\chand\Documents\GitHub\Thesis\M4-1.6 mM\RT\Raw_data\merged_pressure_log.csv")
    sample_df = sample_df[sample_df["Elapsed Time (hours)"] > 4.0]
    sample_df = sample_df[sample_df["Elapsed Time (hours)"] <= 310]

    sample_time = sample_df["Elapsed Time (hours)"].values
    sample_signal = sample_df["MPR-Sch2 (kPa)"].values
    sample_temp = sample_df["Temperature (C)"].values
    sample_date = sample_df["Date"].values
    sample_clock = sample_df["Time"].values
    return sample_time, sample_signal, sample_temp, sample_date, sample_clock

# === 2. Manual Baseline Correction (with ±0.25 h local averaging) ===
def preprocess_signals_manual_baseline(sample_time, sample_signal,
                                       baseline_start=4, baseline_end=6,
                                       avg_window_h=0.2):
    # restrict to baseline search window
    mask = (sample_time >= baseline_start) & (sample_time <= baseline_end)
    sub_time = sample_time[mask]
    sub_signal = sample_signal[mask]
    if len(sub_signal) < 10:
        raise ValueError("Too few points for baseline detection.")

    # find local minimum in that region
    idx_min = np.argmin(sub_signal)
    t_min = sub_time[idx_min]

    # average ±avg_window_h around that minimum
    mask_local = (sample_time >= t_min - avg_window_h) & (sample_time <= t_min + avg_window_h)
    baseline = np.mean(sample_signal[mask_local])

    corrected_signal = sample_signal - baseline
    print(f"Baseline center {t_min:.2f} h, mean in ±{avg_window_h} h window = {baseline:.4f} kPa")

    return corrected_signal, baseline, t_min

# === 3. Uniform Time Grid Creation ===
def create_uniform_grid(sample_time, sample_signal, corrected_signal):
    num_points = int((sample_time.max() - sample_time.min()) * 3600)
    uniform_time = np.linspace(sample_time.min(), sample_time.max(), num=num_points)
    interp = interp1d(sample_time, sample_signal, kind='linear', fill_value='extrapolate')
    interp_corr = interp1d(sample_time, corrected_signal, kind='linear', fill_value='extrapolate')
    return uniform_time, interp(uniform_time), interp_corr(uniform_time)

# === 4. Savitzky-Golay Smoothing ===
def smooth_signal(signal, window=13, polyorder=3):
    return savgol_filter(signal, window_length=window, polyorder=polyorder)

# === 5. DWT Denoising ===
def dwt_denoise(signal, wavelet='db4', keep_levels=(3, 4, 5)):
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(signal), wavelet_obj.dec_len)
    coeffs = pywt.wavedec(signal, wavelet_obj, level=max_level)
    for i in range(1, len(coeffs)):
        if i not in keep_levels:
            coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wavelet_obj)[:len(signal)]

# === 6. First-Order Model Fit ===
def first_order_model(t, P_max, k):
    return P_max * (1 - np.exp(-k * t))

def fit_first_order_model(time, signal):
    Pmax = np.max(signal)
    k_init = np.log(2) / 35.8
    popt, _ = curve_fit(first_order_model, time, signal, p0=[Pmax, k_init])
    return popt, first_order_model(time, *popt)

def logistic_model(t, P0, k, P_max):
    return P_max / (1 + np.exp(-k * (t - P0)))

def fit_logistic_model(time, signal):
    P_max = np.max(signal)
    P0_init = time[np.argmin(np.abs(signal - P_max / 2))]
    k_init = 1 / time[-1]
    n = len(signal)

    bounds = [(P0_init * 0.8, P0_init * 1.2), (0, 1), (P_max * 0.8, P_max * 1.2)]
    res = minimize(
        lambda p: mean_squared_error(signal, logistic_model(time, *p)),
        [P0_init, k_init, P_max],
        bounds=bounds
    )
    popt = res.x
    fit = logistic_model(time, *popt)
    return popt, fit

# # === 7. Plot with Temperature Coloring ===
# def plot_colored_line(ax, x, y, temp_data, cmap='turbo', linewidth=2.5):
#     norm = Normalize(vmin=np.percentile(temp_data, 1), vmax=np.percentile(temp_data, 99))
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, cmap=cmap, norm=norm, array=temp_data, linewidth=linewidth)
#     ax.add_collection(lc)
#     return lc, norm

def aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

def bic(n, rss, k):
    return n * np.log(rss / n) + k * np.log(n)

# === MAIN EXECUTION ===
sample_time, sample_signal, sample_temp, sample_date, sample_clock = load_and_prepare_data()
corrected_signal, baseline, baseline_time = preprocess_signals_manual_baseline(sample_time, sample_signal)
uniform_time, u_sample, u_corrected = create_uniform_grid(sample_time, sample_signal, corrected_signal)
u_corrected_smooth = smooth_signal(u_corrected, window=11)
dwt_signal = dwt_denoise(u_corrected_smooth)

# Interpolate temperature
temp_interp = interp1d(sample_time, sample_temp, kind='linear', fill_value='extrapolate')
temperature_aligned = temp_interp(uniform_time[:len(dwt_signal)])

# Synchronize lengths
L = min(len(uniform_time), len(dwt_signal), len(temperature_aligned))
uniform_time = uniform_time[:L]
dwt_signal = dwt_signal[:L]
temperature_aligned = temperature_aligned[:L]

# Fit first-order model
popt, fitted_model = fit_first_order_model(uniform_time, dwt_signal)
# Logistic model fit
logi_params, logistic_fit = fit_logistic_model(uniform_time, dwt_signal)

# Compute metrics
n = len(dwt_signal)
rss_first = np.sum((dwt_signal - fitted_model) ** 2)
rss_logi = np.sum((dwt_signal - logistic_fit) ** 2)

metrics = pd.DataFrame({
    "Model": ["First-order", "Logistic"],
    "RMSE": [
        np.sqrt(mean_squared_error(dwt_signal, fitted_model)),
        np.sqrt(mean_squared_error(dwt_signal, logistic_fit))
    ],
    "R²": [
        r2_score(dwt_signal, fitted_model),
        r2_score(dwt_signal, logistic_fit)
    ],
    "AIC": [
        aic(n, rss_first, 2),
        aic(n, rss_logi, 3)
    ],
    "BIC": [
        bic(n, rss_first, 2),
        bic(n, rss_logi, 3)
    ],
    "Half-Life (h)": [
        np.log(2) / popt[1],
        logi_params[0]
    ]
})

print("\n=== Model Fit Summary ===")
print(metrics.round(4).to_string(index=False))

# === temp color mapping ===
def plot_colored_line(ax, x, y, temperature_data, cmap='turbo', label=None, linewidth=4, zorder=2, norm=None):
    # Clip to reduce outliers (1st–99th)
    lower_clip = np.percentile(temperature_data, 1)
    upper_clip = np.percentile(temperature_data, 99)
    temp_clipped = np.clip(temperature_data, lower_clip, upper_clip)

    # Only compute norm if not supplied
    if norm is None:
        norm = Normalize(vmin=lower_clip, vmax=upper_clip)

    # Create segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        array=temp_clipped, linewidth=linewidth, zorder=zorder)
    ax.add_collection(lc)

    return lc, norm

# === Plotting ===
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
legend_font = 13
dpi = 900

# Define custom colors
colors = {
    "raw": "#009F82",        # Teal blue
    "drift": "#D95F02",      # Classic orange
    "corrected": "#7570B3",  # Soft purple (keeps consistency)
    "sg": "#E7298A",         # Magenta pink for SG-filtered

    "dwt": '#2c7bb6',        # Dark Blue
    "first": '#e66101',      # Warm Orange – mid-performing
    "logi": '#d95f02',       # Deep Red
    "gomp": "#009F82",       # Teal blue

    "dual": "#999999"        # Gray (less prominent)
}

def format_ax(ax, xlabel="Elapsed time (h)", ylabel="Pressure (kPa)"):
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=15)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
    ax.minorticks_on()
    ax.grid(axis='y', linestyle='--', alpha=0.8)


# ***********************************************************************************
# O2 mole conversion
# ***********************************************************************************

def plot_o2_release_rt(uniform_time, dwt_signal, temperature_aligned,
                       dpi=900):

    os.makedirs(plot_dir, exist_ok=True)

    # === Compute O₂ release (µmol) ===
    V_headspace_L = 0.004
    R = 0.082057
    pressure_atm = dwt_signal / 101.325
    temperature_K = temperature_aligned + 273.15
    o2_umol = ((pressure_atm * V_headspace_L) / (R * temperature_K)) * 1e6

    # --- Temperature-colored line utility ---
    def plot_colored_line(ax, x, y, temp_data, cmap='turbo', linewidth=3.5, norm=None, zorder=3):
        norm = norm or Normalize(vmin=np.percentile(temp_data, 1),
                                 vmax=np.percentile(temp_data, 99))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, array=temp_data,
                            linewidth=linewidth, zorder=zorder)
        ax.add_collection(lc)
        return lc, norm

    # === Main plot ===
    fig, ax = plt.subplots(figsize=(6.75, 5), dpi=dpi)

    # --- Temperature-colored O₂ release curve ---
    lc, shared_temp_norm = plot_colored_line(
        ax, uniform_time, o2_umol, temperature_aligned,
        cmap='turbo', linewidth=3.75
    )

    # --- Colorbar ---
    cbar = fig.colorbar(ScalarMappable(norm=shared_temp_norm, cmap='turbo'), ax=ax, pad=0.01)
    cbar.set_label("Temperature (°C)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.25))
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # --- Axis formatting ---
    ax.set_xlabel("Elapsed Time (h)", fontsize=18)
    ax.set_ylabel("$O_2$ released (µmol)", fontsize=18)
    ax.set_xlim(uniform_time.min(), uniform_time.max())
    ax.set_ylim(0, np.max(o2_umol) * 1.05)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=18)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
    ax.minorticks_on()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Legend (gradient line only) ---
    from matplotlib.legend_handler import HandlerBase

    class HandlerGradientLine(HandlerBase):
        def __init__(self, cmap, norm, lw=3, **kw):
            super().__init__(**kw)
            self.cmap, self.norm, self.lw = cmap, norm, lw

        def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
            n = 256
            gradient = np.linspace(self.norm.vmin, self.norm.vmax, n).reshape(1, -1)
            lw_rel = (self.lw / fontsize) * height
            ymid = y0 + 0.5 * height
            im = plt.imshow(gradient, cmap=self.cmap, norm=self.norm,
                            extent=[x0, x0 + width, ymid - 0.5 * lw_rel, ymid + 0.5 * lw_rel],
                            transform=trans, aspect="auto")
            return [im]

    gradient_proxy = object()
    ax.legend(
        [gradient_proxy],
        ["$O_2$ release profile (Temp-mapped)"],
        handler_map={object: HandlerGradientLine(cmap=plt.cm.turbo,
                                                 norm=shared_temp_norm, lw=4)},
        fontsize=13, frameon=False, loc="lower right"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "IEEEI2MTC_M4_RT_mpr_temp_colored_O2_release.png"), dpi=dpi)
    plt.close()

    print(f"[Saved] $O_2$ release plot → {os.path.join(plot_dir, 'A_IEEEI2MTC_M4_RT_mpr_temp_colored_O2_release.png')}")
    return o2_umol


# === Run ===
o2_umol = plot_o2_release_rt(uniform_time, dwt_signal, temperature_aligned,
                             dpi=dpi)

# ***********************************************************************************
# === Prepare O2 release export table ===
# Constants
V_solution_L = 0.006
V_headspace_L = 0.004
soln_headspace_ratio = V_solution_L / V_headspace_L
n_theoretical_umol = 9.6  # total theoretical µmol O2 expected


# Construct DataFrame
df_o2_export = pd.DataFrame({
    "Sample": ["1.6 mM @ RT"] * len(uniform_time),
    "Time (h)": uniform_time,
    "MPR DWT denoised pressure (kPa)": dwt_signal,
   
    "calibrated temperature (K)": temperature_aligned + 273.15,
    "MPR O2 Released (µmol)": o2_umol,
    
    "MPR % Conversion": (o2_umol / n_theoretical_umol) * 100,
   
    "Soln/Headspace Ratio": [soln_headspace_ratio] * len(uniform_time)
})

# Export
df_o2_export.to_csv(
    os.path.join(csv_output_dir, "M4_1.6mM_RT_o2_release_MPR.csv"),
    index=False
)

print(f"[Saved] Exported O2 release data → {os.path.join(csv_output_dir, 'M4_1.6mM_RT_o2_release_MPR.csv')}")



# === Plot1: DWT-SG denoised Pressure vs raw data ===
fig1, ax1 = plt.subplots(figsize=(6.75, 5), dpi=dpi)

# === Raw Signal (background)
ax1.plot(uniform_time, u_corrected,
         label="Raw pressure signal",
         color=colors["dual"], lw=2.5, alpha=0.4, linestyle='--', zorder=1)

# === Filtered Signal (foreground, with subtle stroke)
ax1.plot(uniform_time, dwt_signal,
         label="Denoised pressure signal, $P(t)$",
         color=colors["dwt"], lw=3, zorder=2,
         path_effects=[withStroke(linewidth=4, foreground='white')])  # optional highlight

# === First-order and Logistic model fits
ax1.plot(uniform_time, fitted_model,
         label="_nolegend_",
         linestyle="", color=colors["gomp"])

ax1.plot(uniform_time, logistic_fit,
         label="_nolegend_",
         linestyle="", color="#d95f02")

format_ax(ax1, ylabel="Pressure (kPa)")
ax1.set_ylim(bottom=0)

# === Legend (auto-detect from plotted lines)
ax1.legend(fontsize=13, frameon=False, loc='lower right')

# === Inset: First few hours ===
inset_ax1 = inset_axes(ax1, width="40%", height="40%", loc="lower right",
                       bbox_to_anchor=(-0.15, 0.22, 1, 1),
                       bbox_transform=ax1.transAxes, borderpad=0)

inset_mask = (uniform_time <= 36)
inset_time = uniform_time[inset_mask]
inset_pressure = dwt_signal[inset_mask]
inset_rawpressure = u_corrected[inset_mask]

# Inset fix
inset_ax1.plot(inset_time, inset_rawpressure,
         color=colors["dual"], lw=2.5, alpha=0.4, linestyle='--', zorder=1)

inset_ax1.plot(inset_time, inset_pressure, color=colors["dwt"], lw=3, zorder=2,
         path_effects=[withStroke(linewidth=4, foreground='white')])

inset_ax1.set_xlim(inset_time.min(), inset_time.max())
inset_ax1.set_ylim(0, inset_pressure.max() * 1.1)
inset_ax1.set_title("First 36 h", fontsize=12)
inset_ax1.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax1.xaxis.set_major_locator(MultipleLocator(10))  # Show ticks every 5 hours
inset_ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
inset_ax1.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
inset_ax1.minorticks_on()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "0_mpr_plot1_M4.png"), dpi=dpi)


# === plot 2. temperature colored pressure profile ===
fig, ax = plt.subplots(figsize=(6.75, 5), dpi=dpi)

# Temperature-colored signal
line, norm = plot_colored_line(ax, uniform_time, dwt_signal, temperature_aligned)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='turbo'), ax=ax, pad=0.01)
cbar.set_label("Temperature (\u00b0C)", fontsize=16)
cbar.ax.tick_params(labelsize=10)
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.25))
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

# Force axis limits based on pressure data only
ax.set_xlim(uniform_time.min(), uniform_time.max())
ax.set_ylim(0, np.max(dwt_signal) * 1.05)

ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=15)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax.minorticks_on()

ax.set_xlabel("Elapsed Time (h)", fontsize=16)
ax.set_ylabel("Pressure (kPa)", fontsize=16)

# === Legend with proxy for LineCollection
pressure_proxy = Line2D([0], [0], color='gray', lw=3, label="Temp-colored denoised pressure signal")
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, pressure_proxy)
labels.insert(0, "Temp-colored denoised pressure signal")
ax.legend(handles, labels, fontsize=13, frameon=False, loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# === Inset: First few hours ===
inset_ax = inset_axes(ax, width="40%", height="40%", loc="lower right",
                      bbox_to_anchor=(-0.15, 0.20, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)

inset_mask = (uniform_time <= 36)
inset_time = uniform_time[inset_mask]
inset_pressure = dwt_signal[inset_mask]
inset_temp = temperature_aligned[inset_mask]

# Use same plotting logic to preserve temperature contrast
line_inset, _ = plot_colored_line(
    inset_ax, inset_time, inset_pressure, inset_temp,
    cmap='turbo', linewidth=3, norm=norm
)

inset_ax.set_xlim(inset_time.min(), inset_time.max())
inset_ax.set_ylim(0, inset_pressure.max() * 1.1)
inset_ax.set_title("First 36 h", fontsize=12)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.xaxis.set_major_locator(MultipleLocator(10))
inset_ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
inset_ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
inset_ax.minorticks_on()

plt.tight_layout()
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, "M4_mpr_temp_colored_pressure_fit.png"), dpi=dpi)


# === Plot2B: Temperature-colored pressure profile with raw overlay (Sample M4) ===
fig, ax = plt.subplots(figsize=(6.75, 5), dpi=dpi)

# --- Raw Signal (gray dashed background)
ax.plot(uniform_time, u_corrected,
        label="Raw experimental signal",
        color="gray", lw=3, alpha=0.3, linestyle="--", zorder=1)

# --- Temperature-colored denoised curve
line, shared_temp_norm = plot_colored_line(
    ax, uniform_time, dwt_signal, temperature_aligned,
    cmap='turbo', label="Temp-colored denoised pressure signal"
)

# --- Colorbar
cbar = fig.colorbar(ScalarMappable(norm=shared_temp_norm, cmap='turbo'), ax=ax, pad=0.01)
cbar.set_label("Temperature (°C)", fontsize=18)
cbar.ax.tick_params(labelsize=14)
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.25))
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

# --- Axes formatting
ax.set_xlabel("Elapsed Time (h)", fontsize=18)
ax.set_ylabel("Pressure (kPa)", fontsize=18)

ax.set_xlim(uniform_time.min(), uniform_time.max())
ax.set_ylim(0, np.max(dwt_signal) * 1.05)

ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=18)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax.minorticks_on()

ax.grid(axis='y', linestyle='--', alpha=0.7)

# --- Legend with turbo gradient line (same as Sample M2 style)
class HandlerGradientLine(HandlerBase):
    def __init__(self, cmap, norm, lw=3, **kw):
        super().__init__(**kw)
        self.cmap = cmap
        self.norm = norm
        self.lw = lw

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        n = 256
        gradient = np.linspace(self.norm.vmin, self.norm.vmax, n).reshape(1, -1)
        lw_rel = (self.lw / fontsize) * height
        ymid = y0 + 0.5 * height
        im = plt.imshow(
            gradient,
            cmap=self.cmap, norm=self.norm,
            extent=[x0, x0+width, ymid - 0.5*lw_rel, ymid + 0.5*lw_rel],
            transform=trans, aspect="auto"
        )
        return [im]

raw_proxy = Line2D([0],[0], color="gray", lw=4, ls="-", alpha=0.4)
gradient_proxy = object()

ax.legend(
    [raw_proxy, gradient_proxy],
    ["Raw signal", "Denoised signal (temp-colored)"],
    handler_map={object: HandlerGradientLine(cmap=plt.cm.turbo,
                                             norm=shared_temp_norm,
                                             lw=4)},
    fontsize=legend_font, frameon=False, loc="lower right"
)

# --- Inset: First 36 h
inset_ax = inset_axes(ax, width="35%", height="35%", loc="lower right",
                      bbox_to_anchor=(-0.14, 0.22, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)

inset_mask = (uniform_time <= 36)
inset_time = uniform_time[inset_mask]
inset_raw = u_corrected[inset_mask]
inset_denoised = dwt_signal[inset_mask]
inset_temp = temperature_aligned[inset_mask]

inset_ax.plot(inset_time, inset_raw, color="gray", lw=2.5, alpha=0.3, linestyle="--", zorder=1)
plot_colored_line(inset_ax, inset_time, inset_denoised, inset_temp,
                  cmap="turbo", linewidth=4, norm=shared_temp_norm, zorder=2)

inset_ax.set_xlim(inset_time.min(), inset_time.max())
inset_ax.set_ylim(0, inset_denoised.max() * 1.1)
inset_ax.set_title("First 36 h", fontsize=14)
inset_ax.grid(axis='y', linestyle='--', alpha=0.5)
inset_ax.xaxis.set_major_locator(MultipleLocator(10))
inset_ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=12)
inset_ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
inset_ax.minorticks_on()

plt.tight_layout()
os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, "ACSOmega_M4_mpr_temp_colored_pressure_fit.png"), dpi=dpi)


# === Plot 3: DWT Pressure Curve with Model Fits ===
fig2, ax2 = plt.subplots(figsize=(6.75, 5), dpi=dpi)
ax2.plot(uniform_time, dwt_signal, label=fr"Denoised pressure signal" + "\n", color=colors["dwt"], lw=3)

first_eq = r"P(t) = P_{\max}(1 - e^{-kt})"
logi_eq = r"P(t) = \frac{P_{\max}}{1 + e^{-k(t - t_0)}}"

ax2.plot(uniform_time, fitted_model,
         label=fr"First-order fit ($R^2$ = {metrics.loc[0, 'R²']:.3f})" + "\n" + f"${first_eq}$" + "\n",
         linestyle="--", color=colors["gomp"], lw=2.5)

ax2.plot(uniform_time, logistic_fit,
         label=fr"Logistic fit ($R^2$ = {metrics.loc[1, 'R²']:.3f})" + "\n" + f"${logi_eq}$",
         linestyle="-", color=colors["logi"], lw=2.25)

format_ax(ax2, ylabel="Pressure (kPa)")
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=13, frameon=True, loc="lower right", facecolor="white", framealpha=0.2)

ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=15)
ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax2.minorticks_on()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "M4_mpr_kinetic_model_fits.png"), dpi=dpi)

# 3
"""
Temperature-Dependent Kinetic Modeling.
To investigate the influence of temperature variations on the pressure-time behavior of ANT-EPO decomposition, 
we extended our kinetic analysis by incorporating temperature as a covariate into both the first-order and logistic models. 
The temperature-dependent rate constant 𝑘(𝑇) was defined using an Arrhenius-like relation, 𝑘(𝑇)=𝐴⋅exp⁡(−𝐸𝑎𝑅𝑇),
where 𝐴 is the pre-exponential factor, 𝐸𝑎 the activation energy, and 𝑇 the local temperature (in Kelvin) interpolated over time.
This formulation enabled the modeling of pressure evolution with a dynamic 𝑘(𝑇) instead of assuming a static rate. 
Both the temperature-adjusted first-order and logistic models were fitted to the full denoised pressure dataset. 
Model performance was evaluated using RMSE, 𝑅2, AIC metrics. 
The temperature-integrated logistic model demonstrated improved alignment with the observed signal, 
supporting the hypothesis that thermal fluctuations meaningfully modulate the oxygen release kinetics. 
This approach offers a more accurate and physically grounded representation of the reaction dynamics under quasi-isothermal 
but drifting ambient conditions.

Modify the standard first-order model:

𝑃(𝑡)=𝑃max⋅(1−𝑒−𝑘𝑡) into a temperature-aware version, e.g.:

𝑃(𝑡,𝑇)=𝑃max⋅(1−𝑒−𝑘(𝑇)⋅𝑡) with 𝑘(𝑇)=𝐴⋅𝑒−𝐸𝑎/(𝑅𝑇) 
So instead of fitting a constant k, you’ll model k as a function of temperature over time.
"""

# === Temperature-Dependent First-Order and Logistic Fits ===
def temp_dependent_first_order(t, A, Ea, Pmax):
    T_K = temperature_aligned[:len(t)] + 273.15
    exponent = -Ea / (R * T_K)
    exponent = np.clip(exponent, -700, 700)
    k_T = A * np.exp(exponent)
    return Pmax * (1 - np.exp(-k_T * t))

def temp_dependent_logistic(t, t0, A, Ea, Pmax):
    T_K = temperature_aligned[:len(t)] + 273.15
    exponent = -Ea / (R * T_K)
    exponent = np.clip(exponent, -700, 700)
    k_T = A * np.exp(exponent)
    return Pmax / (1 + np.exp(-k_T * (t - t0)))

# Fit both models
n = len(uniform_time)

popt_fo_temp, popt_logi_temp = None, None

try:
    popt_fo_temp, _ = curve_fit(
        temp_dependent_first_order,
        uniform_time,
        dwt_signal,
        p0=[1e3, 20000, np.max(dwt_signal)],
        bounds=([1e0, 5000, 0], [1e6, 80000, 5 * np.max(dwt_signal)]),
        maxfev=20000
    )
    fitted_fo_temp = temp_dependent_first_order(uniform_time, *popt_fo_temp)
    rss_fo = np.sum((dwt_signal - fitted_fo_temp) ** 2)
    rmse_fo = np.sqrt(mean_squared_error(dwt_signal, fitted_fo_temp))
    r2_fo = r2_score(dwt_signal, fitted_fo_temp)
    aic_fo = aic(n, rss_fo, 3)
    bic_fo = bic(n, rss_fo, 3)
except:
    fitted_fo_temp = np.full_like(uniform_time, np.nan)
    rmse_fo = r2_fo = aic_fo = bic_fo = np.nan

try:
    popt_logi_temp, _ = curve_fit(
        temp_dependent_logistic,
        uniform_time,
        dwt_signal,
        p0=[uniform_time[n // 2], 1e3, 20000, np.max(dwt_signal)],
        bounds=([uniform_time[0], 1e0, 5000, 0],
                [uniform_time[-1], 1e6, 80000, 5 * np.max(dwt_signal)]),
        maxfev=20000
    )
    fitted_logi_temp = temp_dependent_logistic(uniform_time, *popt_logi_temp)
    rss_logi = np.sum((dwt_signal - fitted_logi_temp) ** 2)
    rmse_logi = np.sqrt(mean_squared_error(dwt_signal, fitted_logi_temp))
    r2_logi = r2_score(dwt_signal, fitted_logi_temp)
    aic_logi = aic(n, rss_logi, 4)
    bic_logi = bic(n, rss_logi, 4)
except:
    fitted_logi_temp = np.full_like(uniform_time, np.nan)
    rmse_logi = r2_logi = aic_logi = bic_logi = np.nan

# === Plot ===
fig, ax = plt.subplots(figsize=(6.75, 5), dpi=dpi)
ax.plot(uniform_time, dwt_signal, label=fr"Denoised pressure signal" + "\n", color=colors["dwt"], lw=2.75)

# LaTeX equations for legend
first_eq_temp = r"P(t, T) = P_{\max}(1 - e^{-k(T).t})"
logi_eq_temp = r"P(t, T) = \frac{P_{\max}}{1 + e^{-k(T).(t - t_0)}}"

ax.plot(uniform_time[::150], fitted_fo_temp[::150],
        label=fr"First-order Fit ($R^2$ = {r2_fo:.3f})" + "\n" + f"${first_eq_temp}$" + "\n",
        linestyle="--", color=colors["sg"], lw=2.25)

ax.plot(uniform_time, fitted_logi_temp,
        label=fr"Logistic Fit ($R^2$ = {r2_logi:.3f})" + "\n" + f"${logi_eq_temp}$",
        linestyle="-", color=colors["logi"], lw=2.25)

format_ax(ax, ylabel="Pressure (kPa)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=13, frameon=True, loc="lower right", facecolor="white", framealpha=0.2)

ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=15)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax.minorticks_on()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "M4_mpr_temp_dependent_kinetic_model_fits.png"), dpi=dpi)

# === Metrics Summary ===
metrics_temp_dep = pd.DataFrame({
    "Model": ["Temp-Dep First-order", "Temp-Dep Logistic"],
    "RMSE": [rmse_fo, rmse_logi],
    "R²": [r2_fo, r2_logi],
    "AIC": [aic_fo, aic_logi],
    "BIC": [bic_fo, bic_logi],
    "Half-Life (h)": [np.nan, popt_logi_temp[0] if popt_logi_temp is not None and len(popt_logi_temp) > 0 else np.nan]
})

print("\n=== Temperature-Dependent Kinetic Model Fit Summary ===")
print(metrics_temp_dep.round(4).to_string(index=False))

# === Wavelet Trend Decomposition (A5 + D5 to D3) ===
# === Perform DWT decomposition (from smoothed signal)
wavelet_name = 'db4'
signal = u_corrected_smooth  # Input your smoothed signal here
uniform_time = uniform_time[:len(signal)]

max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
coeffs = pywt.wavedec(signal, wavelet_name, level=max_level)

# Extract individual levels
reconstructed = {}
levels_to_plot = {"A5": 0, "D5": 1, "D4": 2, "D3": 3}
for label, idx in levels_to_plot.items():
    temp_coeffs = [np.zeros_like(c) for c in coeffs]
    temp_coeffs[idx] = coeffs[idx]
    reconstructed[label] = pywt.waverec(temp_coeffs, wavelet_name)[:len(signal)]

# === Plotting
fig, axs = plt.subplots(len(reconstructed) + 1, 1, figsize=(7, 8), sharex=True, dpi=900)

# Top panel: Original smoothed signal
axs[0].plot(uniform_time, signal, color=colors["sg"], lw=2.5)
axs[0].set_ylabel("SG smoothed\npressure signal", fontsize=15)
axs[0].set_title("Discrete wavelet-Based Trend Decomposition (db4) - multi-levels", fontsize=15)
axs[0].tick_params(axis='both', which='major', direction='in', labelsize=15)
axs[0].minorticks_on()
axs[0].grid(axis='y', linestyle='--', alpha=0.8)

# Individual components
for i, (label, sig) in enumerate(reconstructed.items(), start=1):
    axs[i].plot(uniform_time, sig, lw=2.75, color=f"C{i}", label=label)
    axs[i].set_ylabel(label, fontsize=18)
    axs[i].legend(loc='upper right', fontsize=13)
    axs[i].tick_params(axis='both', which='major', direction='in', labelsize=13)
    axs[i].minorticks_on()
    axs[i].grid(axis='y', linestyle='--', alpha=0.6)

# Final X-label
axs[-1].set_xlabel("Elapsed Time (h)", fontsize=22)

# Optional: tighter y limits
for ax in axs:
    ax.tick_params(axis='both', which='minor', direction='in', length=3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "M4_mpr_wavelet_decomposition.png"), dpi=dpi)


# ==========================================
# === Adaptive Savitzky-Golay ===
# ==========================================
# === 4. visualize the detected 10-hour noise window after applying the variance-based detection ===
def detect_noise_region(signal, window_hours, time_vector, min_time=120, sampling_rate=3600):
    window_size = int(window_hours * sampling_rate)
    valid_indices = np.where(time_vector > min_time)[0]

    if len(valid_indices) < window_size:
        raise ValueError("Not enough data after min_time for noise detection.")

    search_signal = signal[valid_indices]
    rolling_var = pd.Series(search_signal).rolling(window_size).var()
    relative_start_idx = rolling_var.idxmin()

    noise_start_idx = valid_indices[relative_start_idx]
    noise_mask = np.zeros_like(signal, dtype=bool)
    noise_mask[noise_start_idx:noise_start_idx + window_size] = True

    return noise_mask, noise_start_idx, window_size

def safe_normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series))
    return (series - min_val) / (max_val - min_val)

def compute_snr(signal, noise_region):
    signal_p2p = np.ptp(signal)
    noise_rms = np.sqrt(np.mean(noise_region**2))
    if noise_rms == 0:
        return np.inf
    return 20 * np.log10(signal_p2p / noise_rms)

def evaluate_sg_windows(signal, reference_signal, noise_mask, polyorder=3, min_win=11, max_win=99, step=2):
    results = []
    for window in range(min_win, max_win + 1, step):
        if window >= len(signal):
            continue
        smoothed = savgol_filter(signal, window, polyorder)
        noise_std = np.std(smoothed[noise_mask])
        dynamic_prominence = max(0.005, 3 * noise_std)
        peaks, _ = find_peaks(smoothed, prominence=dynamic_prominence)
        snr = compute_snr(smoothed, smoothed[noise_mask])
        rmse = np.sqrt(mean_squared_error(reference_signal, smoothed))
        results.append({"window": window, "snr": snr, "rmse": rmse, "peaks": len(peaks)})

    df = pd.DataFrame(results)
    df['snr_norm'] = safe_normalize(df['snr'])
    df['rmse_norm'] = safe_normalize(df['rmse'])
    df['score'] = 0.6 * df['snr_norm'] + 0.4 * (1 - df['rmse_norm'])
    best_row = df.loc[df['score'].idxmax()]
    return df, best_row

# Force noise detection after 200h
noise_mask, start_idx, window_size = detect_noise_region(u_corrected, window_hours=10, time_vector=uniform_time, min_time=120)
start_time = uniform_time[start_idx]
end_time = uniform_time[start_idx + window_size - 1]
print(f"Adjusted Noise Region: {start_time:.2f} h to {end_time:.2f} h")

results_df, best_window_info = evaluate_sg_windows(
    signal=u_corrected,
    reference_signal=u_corrected,
    noise_mask=noise_mask
)
optimal_window = int(best_window_info["window"])

# === Plot 6: Scoring vs SG Window Size ===
fig_score, ax_score = plt.subplots(figsize=(6.75, 5), dpi=dpi)

# === score formula ===
score_formula = r" = $0.6 \cdot \text{SNR}_\mathrm{norm} + 0.4 \cdot (1 - \text{RMSE}_\mathrm{norm})$"

# Plot score vs. window
# ax_score.plot(results_df["window"], results_df["score"], 
#               color=colors["sg"], lw=2, linestyle='--', marker='s', markersize=8, markeredgewidth=1.2, markeredgecolor=colors["sg"],
#               label=fr"Score" + score_formula + "\n" + "\n" + "Scoring function" + "\n")

ax_score.plot(results_df["window"], results_df["score"], 
              color='#7570b3', lw=2.5, linestyle='-',
              label=fr"Score" + score_formula + "\n" + "\n" + "Scoring function" + "\n")


ax_score.axvline(optimal_window, color='gray', linestyle='--', lw=2, label=f"Optimal window = {optimal_window}")

ax_score.set_xlabel("Savitzky–Golay Window Size", fontsize=16)
ax_score.set_ylabel("Score", fontsize=16)
ax_score.set_title("Adaptive Scoring for SG Filter Optimization", fontsize=16, pad=12)
ax_score.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=13)
ax_score.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
ax_score.minorticks_on()
ax_score.grid(axis='y', linestyle='--', alpha=0.6)

ax_score.legend(fontsize=13, frameon=True, loc='upper right', facecolor="white", framealpha=0.2)


plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "M4_mpr_sg_window_score_plot.png"), dpi=dpi)



# === benchmark ===
def moving_average(signal, window):
    return np.convolve(signal, np.ones(window) / window, mode='same')

def gaussian_smooth(signal, window, sigma_ratio=0.25):
    sigma = window * sigma_ratio
    return gaussian_filter1d(signal, sigma=sigma, mode='nearest', truncate=3.0)

#=== Create a benchmarking function for all filters ===
def compare_smoothing_methods(signal, reference_signal, noise_mask, windows=[15, 25, 41, 51]):
    records = []
    for w in windows:
        if w >= len(signal) or w < 5:
            continue

        # SG
        try:
            sg = savgol_filter(signal, w, polyorder=3)
            snr_sg = compute_snr(sg, sg[noise_mask])
            rmse_sg = np.sqrt(mean_squared_error(reference_signal, sg))
            records.append({"Method": "SG", "Window": w, "SNR": snr_sg, "RMSE": rmse_sg})
        except:
            pass

        # Moving Average
        try:
            ma = moving_average(signal, w)
            snr_ma = compute_snr(ma, ma[noise_mask])
            rmse_ma = np.sqrt(mean_squared_error(reference_signal, ma))
            records.append({"Method": "Moving Avg", "Window": w, "SNR": snr_ma, "RMSE": rmse_ma})
        except:
            pass

        # Gaussian
        try:
            gauss = gaussian_smooth(signal, w)
            snr_g = compute_snr(gauss, gauss[noise_mask])
            rmse_g = np.sqrt(mean_squared_error(reference_signal, gauss))
            records.append({"Method": "Gaussian", "Window": w, "SNR": snr_g, "RMSE": rmse_g})
        except:
            pass

    df = pd.DataFrame(records)
    return df

# === benchmarking function ===
smoothing_comparison_df = compare_smoothing_methods(
    u_corrected, u_corrected, noise_mask, windows=[11, 13, 19, 25, 31, 41, 51, 61, 71]
)
print("\n=== Smoothing Comparison Table (SI Table S1) ===")
print(smoothing_comparison_df.round(5).to_string(index=False))

# ==== plot 7 ====
def plot_snr_rmse_comparison(df, optimal_window=None, save_path="M4_mpr_SI_SNR_RMSE_vs_Window.png"):
    fig, axs = plt.subplots(2, 1, figsize=(6.75, 6), dpi=dpi, sharex=True)

    color_map = {
        "SG": "#E7298A",          # Magenta
        "Moving Avg": "#1B9E77",  # Teal
        "Gaussian": "#7570B3"     # Purple
    }

    marker_map = {
        "SG": "o",
        "Moving Avg": "s",
        "Gaussian": "D"
    }

    # === Plot SNR and RMSE ===
    for method in df["Method"].unique():
        subset = df[df["Method"] == method]
        axs[0].plot(subset["Window"], subset["SNR"],
                    label=method,
                    marker=marker_map.get(method, 'o'),
                    linestyle='-',
                    linewidth=2.25,
                    markersize=6,
                    color=color_map.get(method, "black"))

        axs[1].plot(subset["Window"], subset["RMSE"],
                    label=method,
                    marker=marker_map.get(method, 'o'),
                    linestyle='-',
                    linewidth=2.25,
                    markersize=6,
                    color=color_map.get(method, "black"))

    # === Aesthetics: SNR subplot ===
    axs[0].set_ylabel("SNR (dB)", fontsize=16)
    # axs[0].set_title("SNR vs. Smoothing Window", fontsize=16, pad=12)
    axs[0].legend(fontsize=12, frameon=True, loc='best', facecolor="white", framealpha=0.2)
    axs[0].grid(axis='y', linestyle='--', alpha=0.6)
    axs[0].tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=13)
    axs[0].tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
    axs[0].minorticks_on()

    # === Aesthetics: RMSE subplot ===
    axs[1].set_xlabel("Window Size", fontsize=16)
    axs[1].set_ylabel("RMSE (kPa)", fontsize=16)
    # axs[1].set_title("RMSE vs. Smoothing Window", fontsize=16, pad=12)
    axs[1].legend(fontsize=12, frameon=True, loc='best', facecolor="white", framealpha=0.2)
    axs[1].grid(axis='y', linestyle='--', alpha=0.6)
    axs[1].tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=13)
    axs[1].tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
    axs[1].minorticks_on()

    # Optional vertical line for optimal SG window
    if optimal_window is not None:
        for ax in axs:
            ax.axvline(optimal_window, color='gray', linestyle='', lw=2, label=f"Optimal SG window = {optimal_window}")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, save_path), dpi=dpi)
    plt.close()

plot_snr_rmse_comparison(smoothing_comparison_df, optimal_window=optimal_window)

print("Plot saved successfully")