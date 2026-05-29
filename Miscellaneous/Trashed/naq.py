# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path


# csv_path = "M2_2.18mM_pressure_o2_release.csv"   
# time_col = "Time (s)"
# press_col = "DWT denoised pressure (kPa)"


# # Constants (room temp water)
# ATM_KPA = 101.325           # kPa per atm
# kH = 1.3e-3                 # mol/L/atm  (Henry's for O2 in water ~RT)
# Vsol_L = 0.006              # L (your solution volume)

# def compute_n_aq_umol_from_gauge_kpa(p_kpa: pd.Series) -> pd.Series:

    
#     P_atm = pd.to_numeric(p_kpa) / ATM_KPA
#     n_aq_umol = kH * P_atm * Vsol_L * 1e6
#     return n_aq_umol

# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]
# time_h = df[time_col].astype(float)
# n_aq_umol = compute_n_aq_umol_from_gauge_kpa(df[press_col].astype(float))


# df_out = pd.DataFrame({
#     "Elapsed Time (h)": time_h,
#     "n_aq (µmol)": n_aq_umol
# })

# print(max(n_aq_umol))

# plt.figure(figsize=(7.2, 4.5))
# plt.plot(df_out["Elapsed Time (h)"], df_out["n_aq (µmol)"], linewidth=2)
# plt.xlabel("Elapsed Time (h)")
# plt.ylabel("O₂ released (µmol)  — dissolved (n_aq)")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()




#============================================================================================================
#comparison of headspace and dissolved 

# import pandas as pd 
# import matplotlib.pyplot as plt

# # ------------------ inputs ------------------
# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    # already in hours (per your note)
# press_col = "DWT denoised pressure (kPa)" # gauge kPa
# gas_col   = "O2 Released (µmol)"          # headspace O2 from your CSV
# n_theory  = 13.08                         # µmol (theoretical)
# # -------------------------------------------

# # Constants (room temp water)
# ATM_KPA = 101.325        # kPa per atm
# kH = 1.3e-3              # mol/L/atm (O2 in water ~RT)
# Vsol_L = 0.006           # L

# def n_aq_umol_from_gauge_kpa(p_kpa: pd.Series) -> pd.Series:
#     P_atm = pd.to_numeric(p_kpa) / ATM_KPA

#     return kH * P_atm * Vsol_L * 1e6  # µmol

# # --- load & clean ---
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]


# time_h = pd.to_numeric(df[time_col], errors="coerce")
# p_kpa  = pd.to_numeric(df[press_col], errors="coerce")
# n_gas_umol = pd.to_numeric(df[gas_col], errors="coerce")

# valid = time_h.notna() & p_kpa.notna() & n_gas_umol.notna()
# time_h, p_kpa, n_gas_umol = [s[valid].reset_index(drop=True) for s in (time_h, p_kpa, n_gas_umol)]

# # Start time at 0 (your column said "Time (s)" but you noted it's already hours)
# time_h = time_h - time_h.iloc[0]

# # --- compute dissolved O2 (absolute) ---
# n_aq_umol_abs = n_aq_umol_from_gauge_kpa(p_kpa)

# # --- also compute change-from-start for visibility ---
# n_aq_umol_rel  = (n_aq_umol_abs - n_aq_umol_abs.iloc[0]).clip(lower=0)
# n_gas_umol_rel = (n_gas_umol   - n_gas_umol.iloc[0]).clip(lower=0)

# # ---------- PLOTS ----------
# plt.figure(figsize=(7.6, 4.8))

# plt.plot(time_h, n_gas_umol,   linewidth=2, label="Headspace O₂ (µmol)")
# plt.plot(time_h, n_aq_umol_abs, linewidth=2, label="Dissolved O₂, nₐq (µmol)")
# plt.axhline(n_theory, linestyle="--", linewidth=1.5, label=f"Theoretical = {n_theory} µmol")

# plt.ylabel("Amount (µmol)")
# plt.xlabel("Elapsed Time (h)")
# plt.grid(True, alpha=0.3)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()

# plt.tight_layout()
# plt.show()


#=================================================================================

# directly plotting without seeing the difference:


import pandas as pd 
import matplotlib.pyplot as plt

# ------------------ inputs ------------------
csv_path  = "M2_2.18mM_pressure_o2_release.csv"
time_col  = "Time (s)"                    
press_col = "DWT denoised pressure (kPa)" # PO2 in kPa (no +ATM)
gas_col   = "O2 Released (µmol)"          # headspace O2 from csv
n_theory  = 13.08                         # µmol (theoretical)
# -------------------------------------------

# Constants (room temp water)
ATM_KPA = 101.325        # kPa per atm
kH = 1.3e-3              # mol/L/atm (O2 in water ~RT)
Vsol_L = 0.006           # L

def n_aq_umol_from_kpa(P_kpa: pd.Series) -> pd.Series:

    P_atm = pd.to_numeric(P_kpa, errors="coerce") / ATM_KPA
    return kH * P_atm * Vsol_L * 1e6  # µmol


df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]


time_h     = pd.to_numeric(df[time_col])
p_kpa      = pd.to_numeric(df[press_col])
n_gas_umol = pd.to_numeric(df[gas_col])



time_h = time_h - time_h.iloc[0]


n_aq_umol_abs = n_aq_umol_from_kpa(p_kpa)
n_tot_umol    = n_gas_umol + n_aq_umol_abs
print (max(n_tot_umol))


plt.figure(figsize=(8.2, 4.8))
plt.plot(time_h, n_tot_umol, linewidth=2.3, label="Total O₂ = n_gas + nₐq (µmol)")
plt.axhline(n_theory, linestyle="--", linewidth=1.6, color="black", label=f"Baseline = {n_theory} µmol")
plt.xlabel("Elapsed Time (h)")
plt.ylabel("Total O₂ (µmol)")
plt.grid(True, alpha=0.3)
plt.xlim(left=0)
ymax = max(n_theory, n_tot_umol.max())
plt.ylim(0, ymax * 1.05)
plt.legend()
plt.tight_layout()
plt.show()


#==============================================================================================
# import pandas as pd
# import matplotlib.pyplot as plt

# # ---------- inputs ----------
# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"
# press_col = "DWT denoised pressure (kPa)"   # gauge kPa
# gas_col   = "O2 Released (µmol)"            # usually Δ headspace O2
# fO2_init  = 0.21                            # 0.21 if air; 0 if N2; 1 if pure O2
# n_theory  = 13.08                           # µmol
# # ----------------------------

# ATM_KPA = 101.325
# R = 0.082057   # L·atm·mol^-1·K^-1
# T = 297.15 # or a fixed 297 K
# Vg = 0.002     # L
# kH = 1.3e-3
# Vsol = 0.006



# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]
# t  = pd.to_numeric(df[time_col], errors="coerce").dropna().reset_index(drop=True)
# Pg = pd.to_numeric(df[press_col], errors="coerce").reindex(t.index).astype(float)
# ng = pd.to_numeric(df[gas_col],   errors="coerce").reindex(t.index).astype(float)
# t = t - t.iloc[0]



# # ΔP in atm from gauge kPa
# dP_atm = Pg / ATM_KPA

# # n_aq absolute using PO2 = fO2_init + ΔP (atm)
# n_aq_abs = kH * (fO2_init + dP_atm) * Vsol * 1e6  # µmol
# # Δ n_aq
# dn_aq = n_aq_abs - n_aq_abs.iloc[0]

# # If your "O2 Released (µmol)" is already a delta (most likely), use as-is:
# dn_gas = (ng - ng.iloc[0]).clip(lower=0)

# # Total *released* O2 (what to compare to theory)
# dn_tot = dn_gas + dn_aq


# # geometry factor (µmol/atm)
# A_umol_per_atm = (Vg/(R*T) + kH*Vsol) * 1e6

# # delta P in atm from gauge kPa
# dP_atm = (df["DWT denoised pressure (kPa)"] - df["DWT denoised pressure (kPa)"].iloc[0]) / ATM_KPA

# # predicted total O2 released from pressure only
# n_tot_from_pressure = A_umol_per_atm * dP_atm

# print(f"A = {A_umol_per_atm:.2f} µmol/atm | "
#       f"Final ΔP ≈ {dP_atm.iloc[-1]:.4f} atm | "
#       f"Predicted Δn_tot ≈ {n_tot_from_pressure.iloc[-1]:.2f} µmol")

# print(f"Initial n_aq(abs) = {n_aq_abs.iloc[0]:.3f} µmol (with fO2_init={fO2_init})")
# print(f"Final Δn_gas = {dn_gas.iloc[-1]:.3f} µmol | Final Δn_aq = {dn_aq.iloc[-1]:.3f} µmol")
# print(f"Final Δn_tot = {dn_tot.iloc[-1]:.3f} µmol  "
#       f"({100*dn_tot.iloc[-1]/n_theory:.1f}% of theory {n_theory} µmol)")

# # Plots for verification
# plt.figure(figsize=(8,4.8))
# plt.plot(t, dn_tot,  linewidth=2, label="Δn_tot = Δn_gas + Δn_aq")
# plt.axhline(n_theory, linestyle="--", linewidth=1.5, label=f"Theory = {n_theory} µmol")
# plt.xlabel("Elapsed Time (h)")
# plt.ylabel("Total O₂ released (µmol)")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


#========================================================================================
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # ---------- inputs ----------
# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    # time already in hours
# press_col = "DWT denoised pressure (kPa)" # gauge kPa
# gas_col   = "O2 Released (µmol)"          # headspace O2 (usually cumulative)
# n_theory  = 13.08                         # µmol
# # ----------------------------

# # constants
# ATM_KPA = 101.325            # kPa/atm
# kH = 1.3e-3                  # mol·L^-1·atm^-1 (O2 in water ~RT)
# Vsol_L = 0.006               # L

# # load/clean
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]

# time_h = pd.to_numeric(df[time_col], errors="coerce")
# P_kpa  = pd.to_numeric(df[press_col], errors="coerce")
# n_gas  = pd.to_numeric(df[gas_col],   errors="coerce")

# mask = time_h.notna() & P_kpa.notna() & n_gas.notna()
# time_h, P_kpa, n_gas = time_h[mask].reset_index(drop=True), P_kpa[mask].reset_index(drop=True), n_gas[mask].reset_index(drop=True)

# # start time at 0 h (nice for plotting)
# time_h = time_h - time_h.iloc[0]

# # --- deltas ---
# # ΔP in atm from gauge (relative to t0)
# dP_atm = (P_kpa - P_kpa.iloc[0]) / ATM_KPA

# # Δ n_aq from Henry's law (µmol)
# dn_aq = kH * Vsol_L * dP_atm * 1e6

# # Δ n_gas: if the gas column already starts near 0, treat as delta; else subtract initial
# if abs(n_gas.iloc[0]) < 1e-6:
#     dn_gas = n_gas.copy()
# else:
#     dn_gas = (n_gas - n_gas.iloc[0]).clip(lower=0)

# # total % of theoretical
# dn_tot = dn_gas + dn_aq
# pct_theory = 100.0 * dn_tot / n_theory

# # quick readout
# print(f"Final Δn_gas = {dn_gas.iloc[-1]:.3f} µmol | Final Δn_aq = {dn_aq.iloc[-1]:.3f} µmol")
# print(f"Final Δn_tot = {dn_tot.iloc[-1]:.3f} µmol  ({pct_theory.iloc[-1]:.1f}% of {n_theory} µmol)")

# # plot
# plt.figure(figsize=(8, 4.6))
# plt.plot(time_h, pct_theory, linewidth=2, label="% of experimental O₂ released")
# plt.axhline(100, linestyle="--", linewidth=1.5, label="100% (13.08 µmol)")
# plt.xlabel("Elapsed Time (h)")
# plt.ylabel("% of experimental O₂ released")
# plt.grid(True, alpha=0.3)
# plt.xlim(left=0)
# plt.ylim(bottom=0, top=max(105, np.nanmax(pct_theory)*1.05))
# plt.legend()
# plt.tight_layout()
# plt.show()













#=============================================================================================
# #combined graph considering only o2 inthe pipe
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ------------------ inputs ------------------
# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    # already in hours
# press_col = "DWT denoised pressure (kPa)" # gauge kPa
# gas_col   = "O2 Released (µmol)"          # headspace O2 (µmol)
# n_theory  = 13.08                         # µmol, theoretical max
# # -------------------------------------------

# # Constants (room temp water)
# ATM_KPA = 101.325        # kPa per atm
# kH = 1.3e-3              # mol/L/atm (O2 in water ~RT)
# Vsol_L = 0.006           # L

# def add_n_tot_column(df: pd.DataFrame,
#                      press_col: str,
#                      gas_col: str,
#                      kH: float = 1.3e-3,
#                      Vsol_L: float = 0.006,
#                      ATM_KPA: float = 101.325) -> pd.DataFrame:
#     """
#     Adds:
#       - 'n_aq (µmol)': dissolved O2 from gauge pressure via Henry's law
#       - 'n_tot (µmol)': n_aq + headspace O2 ('O2 Released (µmol)')
#     Returns a copy with the new columns.
#     """
#     out = df.copy()
#     p_kpa = pd.to_numeric(out[press_col], errors="coerce")
#     n_gas = pd.to_numeric(out[gas_col],   errors="coerce")

#     # gauge kPa -> absolute atm
#     p_abs_atm = (p_kpa + ATM_KPA) / ATM_KPA

#     # Henry's law (absolute n_aq, not delta)
#     n_aq_umol = kH * p_abs_atm * Vsol_L * 1e6  # µmol

#     out["n_aq (µmol)"] = n_aq_umol
#     out["n_gas (µmol)"] = n_gas
#     out["n_tot (µmol)"] = out["n_aq (µmol)"] + out["n_gas (µmol)"]
#     return out

# # --- load & clean ---
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]

# for col in [time_col, press_col, gas_col]:
#     if col not in df.columns:
#         raise KeyError(f"Missing column '{col}'. Got: {df.columns.tolist()}")

# time_h = pd.to_numeric(df[time_col], errors="coerce")
# valid  = time_h.notna()
# df = df[valid].reset_index(drop=True)
# time_h = pd.to_numeric(df[time_col], errors="coerce")

# # Shift so time starts at 0 h (keeps timestamps nice)
# elapsed_time_h = time_h - time_h.iloc[0]

# # --- compute n_aq and n_tot ---
# df_calc = add_n_tot_column(df, press_col=press_col, gas_col=gas_col)

# # --- build 4-column output and save ---
# out_df = pd.DataFrame({
#     "Elapsed Time (h)": elapsed_time_h,
#     "n_aq (µmol)": df_calc["n_aq (µmol)"],
#     "O2 Released (µmol)": df_calc["n_gas (µmol)"],
#     "n_tot (µmol)": df_calc["n_tot (µmol)"],
# })

# out_csv_path = Path(csv_path).with_name(Path(csv_path).stem + "_with_totals.csv")
# out_df.to_csv(out_csv_path, index=False)
# print(f"Saved: {out_csv_path}")

# # --- plot n_tot vs time ---
# plt.figure(figsize=(8, 4.6))
# plt.plot(out_df["Elapsed Time (h)"], out_df["n_tot (µmol)"], linewidth=2, label="n_tot = n_aq + n_gas")
# plt.axhline(n_theory, linestyle="--", linewidth=1.5, label=f"Theoretical max = {n_theory} µmol")
# plt.xlabel("Elapsed Time (h)")
# plt.ylabel("n_tot (µmol)")
# plt.grid(True, alpha=0.3)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # --- quick summary in console ---
# final = out_df["n_tot (µmol)"].iloc[-1]
# start = out_df["n_tot (µmol)"].iloc[0]
# print(f"Start n_tot = {start:.3f} µmol | Final n_tot = {final:.3f} µmol")
# print(f"Δ n_tot = {final - start:.3f} µmol | Final vs theory: "
#       f"{final:.3f} / {n_theory} = {100*final/n_theory:.1f}%")



#============================================================================================================

# #comparison plot between all 3 phases but in %

# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ------------------ inputs ------------------
# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    # already in HOURS (just a legacy name)
# press_col = "DWT denoised pressure (kPa)" # gauge kPa
# gas_col   = "O2 Released (µmol)"          # headspace O2 (µmol)
# n_theory  = 13.08                         # µmol, theoretical max (100%)
# # -------------------------------------------

# # Constants (room temp water)
# ATM_KPA = 101.325        # kPa per atm
# kH = 1.3e-3              # mol/L/atm (O2 in water ~RT)
# Vsol_L = 0.006           # L

# def add_n_tot_column(df: pd.DataFrame,
#                      press_col: str,
#                      gas_col: str,
#                      kH: float = 1.3e-3,
#                      Vsol_L: float = 0.006,
#                      ATM_KPA: float = 101.325) -> pd.DataFrame:
#     """
#     Adds absolute amounts (µmol):
#       - 'n_aq (µmol)'  : dissolved O2 via Henry's law from gauge pressure
#       - 'n_gas (µmol)' : headspace O2 (from CSV)
#       - 'n_tot (µmol)' : n_aq + n_gas
#     """
#     out = df.copy()
#     p_kpa = pd.to_numeric(out[press_col], errors="coerce")
#     n_gas = pd.to_numeric(out[gas_col],   errors="coerce")

#     # gauge kPa -> absolute atm
#     p_abs_atm = (p_kpa + ATM_KPA) / ATM_KPA

#     # Henry's law (absolute n_aq)
#     n_aq_umol = kH * p_abs_atm * Vsol_L * 1e6  # µmol

#     out["n_aq (µmol)"]  = n_aq_umol
#     out["n_gas (µmol)"] = n_gas
#     out["n_tot (µmol)"] = out["n_aq (µmol)"] + out["n_gas (µmol)"]
#     return out

# # --- load & clean ---
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]
# for col in [time_col, press_col, gas_col]:
#     if col not in df.columns:
#         raise KeyError(f"Missing column '{col}'. Got: {df.columns.tolist()}")

# time_h = pd.to_numeric(df[time_col], errors="coerce")   # already hours
# valid  = time_h.notna()
# df = df[valid].reset_index(drop=True)
# time_h = pd.to_numeric(df[time_col], errors="coerce")

# # Start time at 0 h
# elapsed_time_h = time_h - time_h.iloc[0]

# # --- compute absolute amounts ---
# df_calc = add_n_tot_column(df, press_col=press_col, gas_col=gas_col)

# # --- make % columns (relative to n_theory) ---
# out_df = pd.DataFrame({
#     "Elapsed Time (h)": elapsed_time_h,
#     "n_aq (µmol)": df_calc["n_aq (µmol)"],
#     "O2 Released (µmol)": df_calc["n_gas (µmol)"],
#     "n_tot (µmol)": df_calc["n_tot (µmol)"],
# })
# out_df["% Dissolved (n_aq)"] = 100.0 * out_df["n_aq (µmol)"] / n_theory
# out_df["% Headspace (O2 Released)"] = 100.0 * out_df["O2 Released (µmol)"] / n_theory
# out_df["% Total (n_tot)"] = 100.0 * out_df["n_tot (µmol)"] / n_theory

# # Save CSV with % columns
# out_csv_path = Path(csv_path).with_name(Path(csv_path).stem + "_with_totals_and_pct_new.csv")
# out_df.to_csv(out_csv_path, index=False)
# print(f"Saved: {out_csv_path}")

# # --- plot all three % on the same axes ---
# plt.figure(figsize=(8.6, 4.8))
# plt.plot(out_df["Elapsed Time (h)"], out_df["% Headspace (O2 Released)"],
#          linewidth=2, label="Headspace O₂ (%)", color="tab:blue")
# plt.plot(out_df["Elapsed Time (h)"], out_df["% Dissolved (n_aq)"],
#          linewidth=2, label="Dissolved O₂ (nₐq) (%)", color="tab:orange")
# plt.plot(out_df["Elapsed Time (h)"], out_df["% Total (n_tot)"],
#          linewidth=2, label="Total O₂ (%)", color="tab:green")

# plt.axhline(100, linestyle="--", linewidth=1.5, color="black", label=f"100% = {n_theory} µmol")
# plt.xlabel("Elapsed Time (h)")
# plt.ylabel("Percent of theoretical O₂ (%)")
# plt.grid(True, alpha=0.3)
# plt.xlim(left=0)
# ymax = max(100, out_df["% Total (n_tot)"].max(),
#            out_df["% Dissolved (n_aq)"].max(),
#            out_df["% Headspace (O2 Released)"].max())
# plt.ylim(0, ymax * 1.05)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # --- quick console summary ---
# final_tot = out_df["% Total (n_tot)"].iloc[-1]
# final_gas = out_df["% Headspace (O2 Released)"].iloc[-1]
# final_aq  = out_df["% Dissolved (n_aq)"].iloc[-1]
# print(f"Final % — Total: {final_tot:.1f}%, Headspace: {final_gas:.1f}%, Dissolved: {final_aq:.1f}%")

#=======================================================================================================

#modelling naq


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # ------------------ inputs ------------------
# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    # NOTE: your file stores HOURS even though name says (s)
# press_col = "DWT denoised pressure (kPa)" # gauge kPa (above atmosphere)
# # -------------------------------------------

# # Constants (room temp water; from your doc)
# ATM_KPA = 101.325                         # kPa per atm
# kH = 1.3e-3                               # mol·L^-1·atm^-1  (O2 in water, ~297 K)
# Vsol_L = 0.006                            # L (solution volume)

# def compute_naq_umol_from_pressure_is_PO2(p_gauge_kpa: pd.Series) -> pd.Series:
#     """
#     Assumes the headspace is pure O2, so absolute pressure equals PO2.
#     Henry's law: n_aq (mol) = kH * P_O2(atm) * Vsol, then convert to µmol.
#     """
#     P_abs_atm = (pd.to_numeric(p_gauge_kpa, errors="coerce") + ATM_KPA) / ATM_KPA
#     n_aq_umol = kH * P_abs_atm * Vsol_L * 1e6  # µmol
#     return n_aq_umol

# def fit_first_order_absolute(time_h: np.ndarray, n_aq_umol: np.ndarray):
#     """
#     Fit: n_aq(t) = n0 + A * (1 - exp(-k t))
#     Grid-search k (no SciPy), closed-form A for each k, return best (n0, A, k, y_fit, R2 on Δ).
#     """
#     t = time_h - time_h[0]
#     n0 = float(n_aq_umol[0])
#     y  = np.clip(n_aq_umol - n0, a_min=0, a_max=None)

#     k_grid = np.logspace(-5, 1, 300)  # 1/h
#     best = None
#     for k in k_grid:
#         m = 1.0 - np.exp(-k * t)
#         denom = float(np.dot(m, m))
#         if denom <= 0:
#             continue
#         A = float(np.dot(m, y) / denom)
#         y_hat = A * m
#         sse = float(np.sum((y - y_hat)**2))
#         if best is None or sse < best[0]:
#             best = (sse, k, A)

#     sse, k_best, A_best = best
#     y_fit_abs = n0 + A_best * (1.0 - np.exp(-k_best * t))

#     ss_tot = float(np.sum((y - y.mean())**2)) if np.any(y) else np.nan
#     R2 = 1.0 - sse/ss_tot if ss_tot and not np.isnan(ss_tot) else np.nan
#     return n0, A_best, k_best, y_fit_abs, R2

# # ---------- load & compute ----------
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]
# if time_col not in df.columns or press_col not in df.columns:
#     raise KeyError(f"Missing needed columns. Found: {df.columns.tolist()}")

# time_h = pd.to_numeric(df[time_col], errors="coerce")  # already HOURS despite name
# p_kpa   = pd.to_numeric(df[press_col], errors="coerce")
# valid   = time_h.notna() & p_kpa.notna()
# time_h, p_kpa = time_h[valid].reset_index(drop=True), p_kpa[valid].reset_index(drop=True)
# time_h = time_h - time_h.iloc[0]  # start at 0 h

# # Dissolved O2 from Henry (treat measured abs pressure as PO2 since pure O2)
# n_aq_umol = compute_naq_umol_from_pressure_is_PO2(p_kpa).astype(float)

# # ---------- fit & plot ----------
# n0, A, k, y_fit_abs, R2 = fit_first_order_absolute(time_h.values, n_aq_umol.values)
# print(f"Baseline n_aq(t0) = {n0:.3f} µmol")
# print(f"Fitted k          = {k:.4f} 1/h")
# print(f"Δn_aq(∞) = A      = {A:.3f} µmol")
# print(f"n_aq(∞) absolute  = {n0 + A:.3f} µmol")
# print(f"R² (on Δn_aq)     = {R2:.3f}")

# plt.figure(figsize=(8.6, 4.8))
# plt.plot(time_h, n_aq_umol, marker='o', markersize=3, linewidth=1.5, label="nₐq data (µmol)")
# plt.plot(time_h, y_fit_abs, linewidth=2.5, label=f"First-order fit (k={k:.4f})")
# plt.xlabel("Elapsed Time (h)")
# plt.ylabel("Dissolved O₂, nₐq (µmol)")
# plt.grid(True, alpha=0.3)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()
# # Zoom the y-axis
# ymax = max(n_aq_umol.max(), y_fit_abs.max())
# plt.ylim(7, ymax * 1.02)   # start at 6 µmol, small headroom on top

# plt.tight_layout()
# plt.show()




# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ------------------ inputs ------------------
# csv_path  = "M2_2.18mM_pressure_o2_release.csv"
# time_col  = "Time (s)"                    # already in hours (legacy name)
# press_col = "DWT denoised pressure (kPa)" # PO2 in kPa (no +ATM)
# gas_col   = "O2 Released (µmol)"          # headspace O2 from your CSV
# n_theory  = 13.08                         # µmol (theoretical)
# # -------------------------------------------

# # Constants (room temp water)
# ATM_KPA = 101.325        # kPa per atm
# kH = 1.3e-3              # mol/L/atm (O2 in water ~RT)
# Vsol_L = 0.006           # L

# def n_aq_umol_from_kpa(p_kpa: pd.Series) -> pd.Series:
#     """Treat pressure column as PO2 in kPa -> atm, then Henry's law to µmol."""
#     P_atm = pd.to_numeric(p_kpa, errors="coerce") / ATM_KPA
#     return kH * P_atm * Vsol_L * 1e6  # µmol

# # --- load & clean ---
# df = pd.read_csv(csv_path)
# df.columns = [c.strip() for c in df.columns]

# time_h     = pd.to_numeric(df[time_col], errors="coerce")
# p_kpa      = pd.to_numeric(df[press_col], errors="coerce")
# n_gas_umol = pd.to_numeric(df[gas_col],  errors="coerce")

# valid = time_h.notna() & p_kpa.notna() & n_gas_umol.notna()
# time_h, p_kpa, n_gas_umol = [s[valid].reset_index(drop=True) for s in (time_h, p_kpa, n_gas_umol)]

# # Start time at 0 (column is already hours)
# time_h = time_h - time_h.iloc[0]

# # --- compute dissolved & TOTAL O2 (absolute) ---
# n_aq_umol_abs = n_aq_umol_from_kpa(p_kpa)
# n_tot_umol    = n_gas_umol + n_aq_umol_abs

# # --- SAVE CSV with parts and total (µmol) ---
# out_df = pd.DataFrame({
#     "Elapsed Time (h)": time_h,
#     "n_gas (µmol)": n_gas_umol,
#     "n_aq (µmol)": n_aq_umol_abs,
#     "n_tot (µmol)": n_tot_umol
# })

# out_path = Path(csv_path).with_name(Path(csv_path).stem + "_parts_and_total_umol.csv")
# out_df.to_csv(out_path, index=False, float_format="%.6f")
# print(f"Saved: {out_path}")

# # ---------- PLOT (optional) ----------
# fig, ax = plt.subplots(figsize=(7.6, 4.8))
# ax.plot(time_h, n_gas_umol,    linewidth=2, label="Headspace O₂ (µmol)")
# ax.plot(time_h, n_aq_umol_abs, linewidth=2, label="Dissolved O₂, nₐq (µmol)")
# ax.plot(time_h, n_tot_umol,    linewidth=2, label="Total O₂ (µmol)")
# ax.axhline(n_theory, linestyle="--", linewidth=1.5, label=f"Theoretical = {n_theory} µmol")
# ax.set_xlabel("Elapsed Time (h)")
# ax.set_ylabel("Amount (µmol)")
# ax.grid(True, alpha=0.3)
# ax.legend()
# plt.tight_layout()
# plt.show()
