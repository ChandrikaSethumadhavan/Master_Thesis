import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# --- Load data ---
data_path = r"C:\Users\chand\Documents\GitHub\Thesis\CSV files outputs\Modified vant hoff modelling\ntot values.csv"
df = pd.read_csv(data_path)
df.columns = [c.strip() for c in df.columns]

# Extract and clean data
t_h = pd.to_numeric(df["Elapsed Time (h)"], errors="coerce") - df["Elapsed Time (h)"].iloc[0]
n_tot = pd.to_numeric(df["n_total (µmol)"], errors="coerce") * 1e-6  # µmol → mol
k_H = pd.to_numeric(df["kH_mol_per_L_atm"], errors="coerce")
T_K = pd.to_numeric(df["T_K"], errors="coerce")
R = 0.082057366079  # L·atm/mol·K

# --- Sidebar sliders ---
st.sidebar.title("Volume Parameters")
Vsol_mL = st.sidebar.slider("Solution Volume (Vsol, mL)", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
Vg_mL   = st.sidebar.slider("Gas Headspace Volume (Vg, mL)", min_value=0.5, max_value=6.0, value=4.0, step=0.1)

# Convert mL to L
Vsol = Vsol_mL / 1000
Vg   = Vg_mL / 1000

# --- Pressure Prediction ---
A_gas = Vg / (R * T_K)
A_aq  = k_H * Vsol
A_total = A_gas + A_aq

P_pred_atm = n_tot / A_total
P_pred_kPa = P_pred_atm * 101.325  # atm → kPa

# --- Plotting ---
st.title("Predicted Pressure Based on Vsol and Vg")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(t_h, P_pred_kPa, lw=2)
ax.set_xlabel("Elapsed Time (h)")
ax.set_ylabel("Predicted Pressure (kPa)")
ax.set_title(f"Pressure Profile | Vsol = {Vsol_mL:.1f} mL, Vg = {Vg_mL:.1f} mL")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# --- Max Pressure Info ---
st.markdown(f"**Maximum Predicted Pressure:** {np.max(P_pred_kPa):.4f} kPa")
st.markdown(f"**Final Predicted Pressure:** {P_pred_kPa.iloc[-1]:.4f} kPa")

# --- Prepare CSV ---
df_export = pd.DataFrame({
    "Elapsed Time (h)": t_h,
    "Temperature (K)": T_K,
    "kH (mol/L/atm)": k_H,
    "n_total (mol)": n_tot,
    "Predicted Pressure (kPa)": P_pred_kPa
})

csv = df_export.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Predicted Pressure CSV",
    data=csv,
    file_name=f"predicted_pressure_Vsol_{Vsol_mL}_Vg_{Vg_mL}.csv",
    mime="text/csv"
)





















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
