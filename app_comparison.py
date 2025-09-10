import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

st.set_page_config(page_title="Pressure Predictor: RT vs 37°C", layout="wide")

# =========================
# LOAD DATA (RT)
# =========================
rt_ntotal_csv = Path("data/ntotal_output_RT.csv")
rt_kh_csv     = Path("data/henry's constants.csv")

df_rt   = pd.read_csv(rt_ntotal_csv)
df_rt.columns = [c.strip() for c in df_rt.columns]
df_rt_kh = pd.read_csv(rt_kh_csv)
df_rt_kh.columns = [c.strip() for c in df_rt_kh.columns]

# parse RT series
t_rt_h   = pd.to_numeric(df_rt["Elapsed Time (h)"], errors="coerce")
ntot_rt  = pd.to_numeric(df_rt["n_total(µmol)"], errors="coerce")         # µmol
kH_rt    = pd.to_numeric(df_rt["kH (mol/L/atm)"], errors="coerce")        # could be constant/series
T_RT     = pd.to_numeric(df_rt_kh["Temperature_K"], errors="coerce")      # series (time-varying)

# align RT lengths
n_rt = min(len(t_rt_h), len(ntot_rt), len(kH_rt), len(T_RT))
t_rt_h, ntot_rt, kH_rt, T_RT = [s.iloc[:n_rt].reset_index(drop=True) for s in (t_rt_h, ntot_rt, kH_rt, T_RT)]

# =========================
# LOAD DATA (37 °C)
# =========================
t37_ntotal_csv = Path("data/ntotal_output_37.csv")
t37_kh_csv     = Path("data/henrys_constants_37deg.csv")

df_37   = pd.read_csv(t37_ntotal_csv)
df_37.columns = [c.strip() for c in df_37.columns]
df_37_kh = pd.read_csv(t37_kh_csv)
df_37_kh.columns = [c.strip() for c in df_37_kh.columns]

# parse 37 series
t_37_h  = pd.to_numeric(df_37["Elapsed Time (h)"], errors="coerce")
ntot_37 = pd.to_numeric(df_37["n_total(µmol)"], errors="coerce")
kH_37   = pd.to_numeric(df_37["kH (mol/L/atm)"], errors="coerce")
T_37    = pd.to_numeric(df_37_kh["Temperature_K"], errors="coerce")

# align 37 lengths
n_37 = min(len(t_37_h), len(ntot_37), len(kH_37), len(T_37))
t_37_h, ntot_37, kH_37, T_37 = [s.iloc[:n_37].reset_index(drop=True) for s in (t_37_h, ntot_37, kH_37, T_37)]

# =========================
# CONSTANTS & SIDEBAR
# =========================
R = 0.082057   # L·atm·mol^-1·K^-1
st.sidebar.title("Geometry")
Vsol_mL = st.sidebar.slider("Solution Volume Vsol (mL)", 1.0, 10.0, 6.0, 0.1)
Vg_mL   = st.sidebar.slider("Headspace Volume Vg (mL)", 0.5, 6.0, 4.0, 0.1)

Vsol = Vsol_mL / 1000.0
Vg   = Vg_mL   / 1000.0

# Volume scaling relative to reference 6 mL (your ntotal files are based on 6 mL experiments)
Vsol_ref = 0.006
volume_ratio = Vsol / Vsol_ref

# ntotal scaled (µmol -> mol)
ntot_rt_scaled_mol  = (ntot_rt  * volume_ratio) * 1e-6
ntot_37_scaled_mol  = (ntot_37  * volume_ratio) * 1e-6


kH_rt_val = kH_rt
kH_37_val = kH_37

# Gas and aqueous "A" factors — A_gas uses time-varying temperature series (element-wise)
A_gas_rt = Vg / (R * T_RT.values)         # mol/atm (vector)
A_aq_rt  = kH_rt_val * Vsol               # mol/atm (scalar)
A_tot_rt = A_gas_rt + A_aq_rt

A_gas_37 = Vg / (R * T_37.values)
A_aq_37  = kH_37_val * Vsol
A_tot_37 = A_gas_37 + A_aq_37

# Predicted pressure (atm, kPa)
P_rt_atm = ntot_rt_scaled_mol / A_tot_rt
P_37_atm = ntot_37_scaled_mol / A_tot_37
P_rt_kPa = P_rt_atm * 101.325
P_37_kPa = P_37_atm * 101.325

# =========================
# LAYOUT
# =========================
st.title("Predicted Headspace Pressure: Room Temperature vs 37 °C")
st.markdown(f"**Volume scaling factor:** {volume_ratio:.3f} (new Vsol / 6 mL reference)")

# Combined plot
fig, ax = plt.subplots(figsize=(10.5, 5.5))
ax.plot(t_rt_h, P_rt_kPa, lw=2, label=f"Room Temp (RT) – max {np.nanmax(P_rt_kPa):.3f} kPa")
ax.plot(t_37_h, P_37_kPa, lw=2, linestyle="--", label=f"37 °C – max {np.nanmax(P_37_kPa):.3f} kPa")
ax.set_xlabel("Elapsed Time (h)")
ax.set_ylabel("Predicted Pressure (kPa)")
ax.set_title(f"Pressure vs Time | Vsol={Vsol_mL:.1f} mL, Vg={Vg_mL:.1f} mL")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
st.pyplot(fig)

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("RT: Max P (kPa)", f"{np.nanmax(P_rt_kPa):.3f}")
with c2:
    st.metric("RT: Final P (kPa)", f"{P_rt_kPa.iloc[-1]:.3f}")
with c3:
    st.metric("37 °C: Max P (kPa)", f"{np.nanmax(P_37_kPa):.3f}")
with c4:
    st.metric("37 °C: Final P (kPa)", f"{P_37_kPa.iloc[-1]:.3f}")

# Info boxes
with st.expander("Details used in predictions"):
    st.write(f"- R = {R} L·atm·mol⁻¹·K⁻¹")
    st.write(f"- Vsol = {Vsol_mL:.1f} mL, Vg = {Vg_mL:.1f} mL")
    st.write(f"- kH (RT) ≈ Modelled with vant hoff according to each temperature value at each timestamp")
    st.write(f"- kH (37 °C) ≈ Modelled with vant hoff according to each temperature value at each timestamp")
    st.write(f"- A_gas computed with time-varying Temperature_K from each dataset (element-wise)")

# =========================
# EXPORT
# =========================
# --- build per-dataset frames ---
rt_df = pd.DataFrame({
    "Time_h": t_rt_h,
    "P_RT_kPa": P_rt_kPa,
    "T_RT_K": T_RT,
})


t37_df = pd.DataFrame({
    "Time_h": t_37_h,
    "P_37_kPa": P_37_kPa,
    "T_37_K": T_37,
})


# --- outer merge on time, keep all rows from both, sorted by time ---
export = (
    pd.merge(rt_df, t37_df, on="Time_h", how="outer")
      .sort_values("Time_h")
      .reset_index(drop=True)
)

# add geometry + scaling columns
export["Vsol_mL"] = Vsol_mL
export["Vg_mL"]   = Vg_mL
export["volume_ratio"] = volume_ratio

st.download_button(
    "📥 Download combined prediction (CSV)",
    export.to_csv(index=False).encode("utf-8"),
    file_name=f"pressure_predictions_RT_vs_37_Vsol{Vsol_mL}_Vg{Vg_mL}.csv",
    mime="text/csv"
)