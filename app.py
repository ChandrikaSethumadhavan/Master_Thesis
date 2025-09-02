import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# --- Load data ---
import os
data_path = os.path.join(os.path.dirname(__file__), "ntot_values.csv")

df = pd.read_csv(data_path)
df.columns = [c.strip() for c in df.columns]

# Extract and clean data
t_h = pd.to_numeric(df["Elapsed Time (h)"], errors="coerce")
n_tot_umol = pd.to_numeric(df["n_total (µmol)"], errors="coerce")  # Keep in µmol for now
k_H = pd.to_numeric(df["kH_mol_per_L_atm"], errors="coerce")

# Constants (as per your specifications)
R = 0.08206  # L·atm/mol·K (as specified in your documents)
T_K = 297    # K (as you requested: "always T = 297 unless mentioned specifically")

# --- Sidebar sliders ---
st.sidebar.title("Volume Parameters")
Vsol_mL = st.sidebar.slider("Solution Volume (Vsol, mL)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
Vg_mL   = st.sidebar.slider("Gas Headspace Volume (Vg, mL)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)

# Convert mL to L
Vsol = Vsol_mL / 1000
Vg   = Vg_mL / 1000

# --- Volume Scaling (BEFORE unit conversion) ---
# Reference volume from your experiment
Vsol_ref = 0.006  # L (6 mL reference)
volume_ratio = Vsol / Vsol_ref

# Scale the experimental ntotal based on volume ratio
ntotal_scaled_umol = n_tot_umol * volume_ratio

# Now convert to mol
ntotal_scaled_mol = ntotal_scaled_umol * 1e-6

# --- Pressure Prediction ---
# Use first value of kH if it's time-varying, or mean if multiple values
kH_value = k_H.iloc[0] if len(k_H.dropna()) > 0 else 1.3e-3  # fallback value

A_gas = Vg / (R * T_K)
A_aq = kH_value * Vsol
A_total = A_gas + A_aq

P_pred_atm = ntotal_scaled_mol / A_total
P_pred_kPa = P_pred_atm * 101.325  # atm → kPa

# --- Plotting ---
st.title("Predicted Pressure Based on Vsol and Vg (Scaled ntotal)")
st.markdown(f"**Volume Scaling Factor:** {volume_ratio:.3f} (New volume / Reference volume)")
st.markdown(f"**Reference Experiment:** 6 mL solution, 2 mL headspace")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_h, P_pred_kPa, lw=2, color='blue', label=f'Predicted ({Vsol_mL:.1f}:{Vg_mL:.1f})')

# Plot reference case for comparison
if volume_ratio != 1.0:
    P_ref_atm = (n_tot_umol * 1e-6) / ((0.002/(R*T_K)) + (kH_value*0.006))
    P_ref_kPa = P_ref_atm * 101.325
    ax.plot(t_h, P_ref_kPa, '--', lw=1.5, color='red', alpha=0.7, label='Reference (6:2)')

ax.set_xlabel("Elapsed Time (h)")
ax.set_ylabel("Predicted Pressure (kPa)")
ax.set_title(f"Pressure Profile | Vsol = {Vsol_mL:.1f} mL, Vg = {Vg_mL:.1f} mL")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# --- Results Summary ---
st.markdown("### Prediction Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Max Pressure", f"{np.max(P_pred_kPa):.4f} kPa")
with col2:
    st.metric("Final Pressure", f"{P_pred_kPa.iloc[-1]:.4f} kPa")
# with col3:
#     st.metric("Max O₂ Scaled", f"{np.max(ntotal_scaled_umol):.2f} µmol")

# --- Physics Check ---
st.markdown("### Physics Validation")
theoretical_max = 2.18 * Vsol * 1000  # µmol theoretical max for this volume
conversion_efficiency = np.max(ntotal_scaled_umol) / theoretical_max * 100

st.markdown(f"**Theoretical Max O₂:** {theoretical_max:.2f} µmol")
st.markdown(f"**Predicted Max O₂:** {np.max(ntotal_scaled_umol):.2f} µmol")
st.markdown(f"**Conversion Efficiency:** {conversion_efficiency:.1f}%")

#validation checks:
# if Vg_mL < 2.0:
#     st.error(" Headspace too small - predictions unreliable")
    

    



# --- Export Data ---
df_export = pd.DataFrame({
    "Elapsed Time (h)": t_h,
    "Temperature (K)": np.full_like(t_h, T_K),
    "kH (mol/L/atm)": np.full_like(t_h, kH_value),
    "n_total_original (µmol)": n_tot_umol,
    "n_total_scaled (µmol)": ntotal_scaled_umol,
    "n_total_scaled (mol)": ntotal_scaled_mol,
    "Volume_ratio": np.full_like(t_h, volume_ratio),
    "Predicted Pressure (atm)": P_pred_atm,
    "Predicted Pressure (kPa)": P_pred_kPa
})

csv = df_export.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Predicted Pressure CSV",
    data=csv,
    file_name=f"predicted_pressure_Vsol_{Vsol_mL}_Vg_{Vg_mL}_scaled.csv",
    mime="text/csv"
)







































# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from io import BytesIO

# # --- Load data ---
# data_path = r"C:\Users\chand\Documents\GitHub\Thesis\CSV files outputs\Modified vant hoff modelling\ntot values.csv"
# df = pd.read_csv(data_path)
# df.columns = [c.strip() for c in df.columns]

# # Extract and clean data
# t_h = pd.to_numeric(df["Elapsed Time (h)"], errors="coerce") - df["Elapsed Time (h)"].iloc[0]
# n_tot = pd.to_numeric(df["n_total (µmol)"], errors="coerce") * 1e-6  # µmol → mol
# k_H = pd.to_numeric(df["kH_mol_per_L_atm"], errors="coerce")
# T_K = pd.to_numeric(df["T_K"], errors="coerce")
# R = 0.082057366079  # L·atm/mol·K

# # --- Sidebar sliders ---
# st.sidebar.title("Volume Parameters")
# Vsol_mL = st.sidebar.slider("Solution Volume (Vsol, mL)", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
# Vg_mL   = st.sidebar.slider("Gas Headspace Volume (Vg, mL)", min_value=0.5, max_value=6.0, value=4.0, step=0.1)

# # Convert mL to L
# Vsol = Vsol_mL / 1000
# Vg   = Vg_mL / 1000

# # --- Pressure Prediction ---
# A_gas = Vg / (R * T_K)
# A_aq  = k_H * Vsol
# A_total = A_gas + A_aq

# P_pred_atm = n_tot / A_total
# P_pred_kPa = P_pred_atm * 101.325  # atm → kPa

# # --- Plotting ---
# st.title("Predicted Pressure Based on Vsol and Vg")
# fig, ax = plt.subplots(figsize=(8.5, 4.8))
# ax.plot(t_h, P_pred_kPa, lw=2)
# ax.set_xlabel("Elapsed Time (h)")
# ax.set_ylabel("Predicted Pressure (kPa)")
# ax.set_title(f"Pressure Profile | Vsol = {Vsol_mL:.1f} mL, Vg = {Vg_mL:.1f} mL")
# ax.grid(True, alpha=0.3)
# st.pyplot(fig)

# # --- Max Pressure Info ---
# st.markdown(f"**Maximum Predicted Pressure:** {np.max(P_pred_kPa):.4f} kPa")
# st.markdown(f"**Final Predicted Pressure:** {P_pred_kPa.iloc[-1]:.4f} kPa")

# # --- Prepare CSV ---
# df_export = pd.DataFrame({
#     "Elapsed Time (h)": t_h,
#     "Temperature (K)": T_K,
#     "kH (mol/L/atm)": k_H,
#     "n_total (mol)": n_tot,
#     "Predicted Pressure (kPa)": P_pred_kPa
# })

# csv = df_export.to_csv(index=False).encode("utf-8")

# st.download_button(
#     label="📥 Download Predicted Pressure CSV",
#     data=csv,
#     file_name=f"predicted_pressure_Vsol_{Vsol_mL}_Vg_{Vg_mL}.csv",
#     mime="text/csv"
# )





















# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Load data ---
# data_path = r"C:\Users\chand\Documents\GitHub\Thesis\CSV files outputs\Modified vant hoff modelling\ntot values.csv"
# df = pd.read_csv(data_path)
# df.columns = [c.strip() for c in df.columns]

# # Extract and clean data
# t_h = pd.to_numeric(df["Elapsed Time (h)"], errors="coerce") - df["Elapsed Time (h)"].iloc[0]
# n_tot = pd.to_numeric(df["n_total (µmol)"], errors="coerce") * 1e-6  # µmol → mol
# k_H = pd.to_numeric(df["kH_mol_per_L_atm"], errors="coerce")
# T_K = pd.to_numeric(df["T_K"], errors="coerce")
# R = 0.082057366079  # L·atm/mol·K

# # --- Sidebar sliders ---
# st.sidebar.title("Volume Parameters")
# Vsol_mL = st.sidebar.slider("Solution Volume (Vsol, mL)", min_value=1.0, max_value=6.0, value=1.0, step=0.1)
# Vg_mL   = st.sidebar.slider("Gas Headspace Volume (Vg, mL)", min_value=1.0, max_value=6.0, value=4.0, step=0.1)

# # Convert mL to L
# Vsol = Vsol_mL / 1000
# Vg   = Vg_mL / 1000

# # --- Pressure Prediction ---
# A_gas = Vg / (R * T_K)
# A_aq  = k_H * Vsol
# A_total = A_gas + A_aq

# P_pred_atm = n_tot / A_total
# P_pred_kPa = P_pred_atm * 101.325  # Convert atm → kPa

# # --- Plotting ---
# st.title("Predicted Pressure Based on Vsol and Vg")
# fig, ax = plt.subplots(figsize=(8.5, 4.8))
# ax.plot(t_h, P_pred_kPa, lw=2)
# ax.set_xlabel("Elapsed Time (h)")
# ax.set_ylabel("Predicted Pressure (kPa)")
# ax.set_title(f"Pressure Profile | Vsol = {Vsol_mL:.1f} mL, Vg = {Vg_mL:.1f} mL")
# ax.grid(True, alpha=0.3)
# st.pyplot(fig)

# # --- Max Pressure Info ---
# st.markdown(f"**Maximum Predicted Pressure:** {np.max(P_pred_kPa):.2f} kPa")
