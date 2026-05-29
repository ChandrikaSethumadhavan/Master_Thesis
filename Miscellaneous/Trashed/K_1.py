import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Load your CSV file
df = pd.read_csv("M2_2.18mM_pressure_o2_release.csv")
clean_df = df.dropna()

# Extract time and pressure
time = df["Time (s)"]
if 'DWT denoised pressure (kPa)' in df.columns:
    df['Pressure (atm)'] = df['DWT denoised pressure (kPa)'] / 101.325
pressure_atm = df["Pressure (atm)"].values



# 2. Logistic fit: y = L / (1 + exp(-k*(t - t0)))
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))






# Fit logistic model
popt_logistic, _ = curve_fit(logistic, time, pressure_atm, 
                            p0=[max(pressure_atm), 0.1, np.median(time)],
                            bounds=([0, 0, 0], [2, 10, max(time)]))
pred_logistic = logistic(time, *popt_logistic)
r2_logistic = r2_score(pressure_atm, pred_logistic)

print(f"\n2. LOGISTIC MODEL: y = L / (1 + exp(-k*(t - t₀)))")
print(f"   Parameters: L = {popt_logistic[0]:.6f} atm, k = {popt_logistic[1]:.6e} s⁻¹, t₀ = {popt_logistic[2]:.2f} s")
print(f"   Rate constant k₂ = {popt_logistic[1]:.6e} s⁻¹")




# Generate smooth time values
t_fit = np.linspace(min(time), max(time), 500)

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(time, pressure_atm, 'ko', label="Raw Data", markersize=3)

plt.plot(t_fit, logistic(t_fit, *popt_logistic), 'g-', linewidth=2, 
         label=f"Logistic Fit (k₂ = {popt_logistic[1]:.2e} s⁻¹)")


plt.xlabel("Time (s)", fontsize=12, fontweight='bold')
plt.ylabel("Pressure (atm)", fontsize=12, fontweight='bold')
plt.title("Pressure (atm) vs Time with Curve Fits", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add text box with rate constants
rate_constants_text = f"""Rate Constants Extracted:
• Logistic: k₂ = {popt_logistic[1]:.2e} s⁻¹  

"""

plt.text(0.02, 0.98, rate_constants_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.show()


# After fitting
slope_max = np.max(np.gradient(pred_logistic, time))
k_from_slope = 4 * slope_max / popt_logistic[0]
print(f"k from algorithm: {popt_logistic[1]:.6e}")
print(f"k from slope: {k_from_slope:.6e}")
#

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from sklearn.metrics import r2_score

# # -----------------------------
# # Load & prep
# # -----------------------------
# df = pd.read_csv("M2_2.18mM_pressure_o2_release.csv").dropna()

# time = df["Time (s)"].values.astype(float)
# if "DWT denoised pressure (kPa)" in df.columns:
#     df["Pressure (atm)"] = df["DWT denoised pressure (kPa)"] / 101.325
# pressure_atm = df["Pressure (atm)"].values.astype(float)

# t_min, t_max = float(time.min()), float(time.max())
# t_med = float(np.median(time))

# # Fixed shared asymptote
# A = 0.05918  # atm  <-- your chosen maximum pressure

# # -----------------------------
# # Models with fixed asymptote
# # -----------------------------
# # Logistic (fixed L=A): y = A / (1 + exp(-k*(t - t0)))
# def logistic_fixed(t, k, t0):
#     return A / (1.0 + np.exp(-k * (t - t0)))

# # Gompertz (fixed A): y = A * exp( -exp( k*(t0 - t) ) )
# def gompertz_fixed(t, k, t0):
#     return A * np.exp(-np.exp(k * (t0 - t)))

# # -----------------------------
# # Fit both models (k, t0 only)
# # -----------------------------
# # Reasonable initials/bounds
# p0_k = 0.02   # tweak if needed
# p0_t0 = t_med

# bounds = ([0.0, 0.0], [10.0, t_max])  # k >= 0, 0 <= t0 <= max time

# popt_log, _  = curve_fit(logistic_fixed,  time, pressure_atm, p0=[p0_k, p0_t0],
#                          bounds=bounds, maxfev=50000)
# popt_gom, _  = curve_fit(gompertz_fixed,  time, pressure_atm, p0=[p0_k, p0_t0],
#                          bounds=bounds, maxfev=50000)

# pred_log  = logistic_fixed(time, *popt_log)
# pred_gomp = gompertz_fixed(time, *popt_gom)

# r2_log  = r2_score(pressure_atm, pred_log)
# r2_gomp = r2_score(pressure_atm, pred_gomp)

# # -----------------------------
# # Inflection points
# # -----------------------------
# # With these parametrizations:
# # Logistic inflection at t0, y = A/2
# # Gompertz inflection at t0, y = A/e
# t0_log, k_log   = popt_log[1], popt_log[0]
# t0_gomp, k_gomp = popt_gom[1], popt_gom[0]

# y_inflect_log  = A / 2.0
# y_inflect_gomp = A / np.e

# # -----------------------------
# # Plot
# # -----------------------------
# t_fit = np.linspace(t_min, t_max, 800)

# plt.figure(figsize=(12, 8))
# plt.plot(time, pressure_atm, "ko", ms=3, label="Raw Data")

# plt.plot(t_fit, logistic_fixed(t_fit, *popt_log), "g-", lw=2,
#          label=f"Logistic (A fixed, R²={r2_log:.3f})")
# plt.plot(t_fit, gompertz_fixed(t_fit, *popt_gom), "b--", lw=2,
#          label=f"Gompertz (A fixed, R²={r2_gomp:.3f})")

# # Asymptote line
# plt.axhline(A, color="gray", ls=":", lw=1, label=f"Asymptote A = {A:.5f} atm")

# # Inflection markers/lines
# plt.axvline(t0_log,  color="g",  ls="--", lw=1)
# plt.scatter([t0_log],  [y_inflect_log],  color="g",  zorder=5, label="Logistic inflection (A/2)")
# plt.axvline(t0_gomp, color="b", ls="--", lw=1)
# plt.scatter([t0_gomp], [y_inflect_gomp], color="b", zorder=5, label="Gompertz inflection (A/e)")

# plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
# plt.ylabel("Pressure (atm)", fontsize=12, fontweight="bold")
# plt.title("Logistic vs Gompertz (Asymptote fixed to 0.05918 atm) with Inflection Points", fontsize=14, fontweight="bold")
# plt.legend(fontsize=10)
# plt.grid(True, alpha=0.3)

# # Parameter textbox
# txt = (
#     "Fixed A (plateau) = 0.05918 atm\n"
#     f"Logistic:  k={k_log:.3e} s^-1, t0={t0_log:.2f} s, y_inflect=A/2={y_inflect_log:.5f} atm\n"
#     f"Gompertz:  k={k_gomp:.3e} s^-1, t0={t0_gomp:.2f} s, y_inflect=A/e={y_inflect_gomp:.5f} atm"
# )
# plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va="top",
#          fontfamily="monospace", fontsize=10,
#          bbox=dict(boxstyle="round", facecolor="w", alpha=0.85))

# plt.tight_layout()
# plt.show()
