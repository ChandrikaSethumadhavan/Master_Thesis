import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Load data ----------
data_path = r"C:\Users\chand\Documents\GitHub\Thesis\CSV files outputs\Modified vant hoff modelling\ntot values.csv"
df = pd.read_csv(data_path)
df.columns = [c.strip() for c in df.columns]

# Extract and clean data
t_h = pd.to_numeric(df["Elapsed Time (h)"], errors="coerce")
t_h = t_h - t_h.iloc[0]                      # start at 0 h
n_tot_umol = pd.to_numeric(df["n_total (µmol)"], errors="coerce")
k_H_series = pd.to_numeric(df["kH_mol_per_L_atm"], errors="coerce")

# ---------- Constants ----------
R   = 0.08206      # L·atm/(mol·K)
T_K = 297          # K (fixed as per your spec)
kH  = k_H_series.dropna().iloc[0] if k_H_series.dropna().size else 1.3e-3

# ---------- Sidebar controls ----------
st.sidebar.title("Sweep Settings (Vsol fixed at 6.0 mL)")
Vsol_mL = 6.0
Vg_min  = st.sidebar.number_input("Vg min (mL)",  min_value=0.5, max_value=6.0, value=2.0, step=0.1)
Vg_max  = st.sidebar.number_input("Vg max (mL)",  min_value=0.5, max_value=6.0, value=3.0, step=0.1)
Vg_step = st.sidebar.number_input("Vg step (mL)", min_value=0.05, max_value=1.0, value=0.1, step=0.05)
show_reference = st.sidebar.checkbox("Show reference (6 mL : 2 mL)", True)

# clamp/prepare sweep values
Vg_values = np.round(np.arange(Vg_min, Vg_max + 1e-9, Vg_step), 2)

# ---------- Precompute terms that don't change with Vg ----------
Vsol_L = Vsol_mL / 1000.0
Vsol_ref_L = 0.006
volume_ratio = Vsol_L / Vsol_ref_L

# Scale ntotal by volume ratio and convert to mol
ntotal_scaled_mol = (n_tot_umol * volume_ratio) * 1e-6

# ---------- Plot all curves ----------
st.title("Pressure predictions for Vg sweep (Vsol fixed at 6.0 mL)")
st.markdown(f"**Vg sweep:** {Vg_min:.2f}–{Vg_max:.2f} mL (step {Vg_step:.2f} mL) &nbsp; | &nbsp; **T:** {T_K} K &nbsp; | &nbsp; **kH:** {kH:.6g} mol·L⁻¹·atm⁻¹")

fig, ax = plt.subplots(figsize=(10, 6))

results = []
for Vg_mL in Vg_values:
    Vg_L = Vg_mL / 1000.0
    A_gas   = Vg_L / (R * T_K)
    A_aq    = kH * Vsol_L
    A_total = A_gas + A_aq
    P_kPa = (ntotal_scaled_mol / A_total) * 101.325

    ax.plot(t_h, P_kPa, lw=1.6, alpha=0.9, label=f"Vg={Vg_mL:.2f} mL")
    results.append({"Vg (mL)": Vg_mL,
                    "Max Pressure (kPa)": float(np.max(P_kPa)),
                    "Final Pressure (kPa)": float(P_kPa.iloc[-1])})

# Optional reference curve (6:2)
if show_reference:
    Vg_ref_L = 0.002
    A_ref = Vg_ref_L / (R*T_K) + kH * 0.006
    P_ref_kPa = ((n_tot_umol * 1e-6) / A_ref) * 101.325
    ax.plot(t_h, P_ref_kPa, "--", lw=2, color="black", label="Reference (6:2)")

# Highlight best (max P)
res_df = pd.DataFrame(results)
best_idx = res_df["Max Pressure (kPa)"].idxmax()
best_Vg  = res_df.loc[best_idx, "Vg (mL)"]

# Replot the best curve thicker for visibility
Vg_L_best = best_Vg / 1000.0
A_best = Vg_L_best / (R*T_K) + kH * Vsol_L
P_best = (ntotal_scaled_mol / A_best) * 101.325
ax.plot(t_h, P_best, lw=3.0, color="tab:red", label=f"Highest P (Vg={best_Vg:.2f} mL)")

ax.set_xlabel("Elapsed Time (h)")
ax.set_ylabel("Predicted Pressure (kPa)")
ax.set_title("Pressure profiles for Vg ∈ [2.0, 3.0] mL (Vsol = 6.0 mL)")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=9)
st.pyplot(fig)

# ---------- Summary metrics ----------
st.markdown("### Summary over sweep")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Vsol (fixed)", f"{Vsol_mL:.1f} mL")
with c2:
    st.metric("Best Vg (by max P)", f"{best_Vg:.2f} mL")
with c3:
    st.metric("Max Pressure at best Vg", f"{res_df['Max Pressure (kPa)'].max():.3f} kPa")

# Show table
st.dataframe(res_df.sort_values("Vg (mL)").reset_index(drop=True))

# ---------- CSV download ----------
csv = res_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download sweep results (CSV)",
    data=csv,
    file_name=f"pressure_sweep_Vsol6_Vg_{Vg_min}-{Vg_max}_step{Vg_step}.csv",
    mime="text/csv"
)
