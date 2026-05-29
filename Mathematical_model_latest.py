  # Not using GPR and Ridge predictions for now since they are very close to the mean_k prediction, can be added later for comparison

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score


CSV_FOLDER = r"C:\Users\chand\Documents\GitHub\Thesis\csv_combi"
TIME_COL     = "Time (h)"
TEMP_COL_K   = "calibrated temperature (K)"
TEMP_COL_37C = "calibrated temperature (K)"   # column is labelled K but stores Celsius for 37C files  // this has to be fixed soon, not a problem for now
PRESS_COL_RT_ABP2  = "ABP2 DWT denoised pressure (kPa)"
PRESS_COL_RT_MPR   = "MPR SG pressure (kPa)"

PRESS_COL_ELEV_ABP2 = "ABP2 SG smoothed pressure (kPa)"
PRESS_COL_ELEV_MPR  = "MPR SG smoothed pressure (kPa)"

O2_COL_ABP2 = "ABP2 O2 Released (µmol)"
O2_COL_MPR  = "MPR O2 Released (µmol)"


TEMP_NOMINAL_K = {"RT": 298.15, "37C": 310.15, "45C": 318.15, "50C": 323.15}
R_J = 8.314         
TEST_SAMPLE_PREFIX = "M6"



def ml_to_m3(ml):
    return ml * 1e-6


def mM_ml_to_mol(C_mM, V_ml):
    return C_mM * V_ml * 1e-6


def get_pressure_col(temp_label, sensor):
    if temp_label == "RT":
        return PRESS_COL_RT_ABP2 if sensor == "ABP2" else PRESS_COL_RT_MPR
    return PRESS_COL_ELEV_ABP2 if sensor == "ABP2" else PRESS_COL_ELEV_MPR


def get_o2_col(sensor):
    return O2_COL_ABP2 if sensor == "ABP2" else O2_COL_MPR


def O2_model_umol(t, eta_hat, C_mM, Vsol_ml, k):
    n_gen_mol = mM_ml_to_mol(C_mM, Vsol_ml)
    return eta_hat * n_gen_mol * (1.0 - np.exp(-k * t)) * 1e6

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return rmse, r2



#Source: Battino et al. 1983, "The Solubility of Oxygen and Ozone in Liquids"
#kH_ref = 1.3e-3 mol/(L·atm) at T_ref = 298.15 K
_battino_T  = np.array([273.15, 278.15, 283.15, 288.15, 293.15,
                         298.15, 303.15, 308.15, 313.15, 318.15,
                         323.15, 328.15, 333.15, 338.15, 343.15, 348.15])

_battino_dH = np.array([-17.60, -16.18, -15.16, -14.15, -13.13,
                         -12.11, -11.10, -10.08,  -9.06,  -8.05,
                          -7.03,  -6.01,  -5.00,  -3.98,  -2.97,  -1.95])
_kH_ref = 1.3e-3      
_T_ref  = 298.15      
_dH_ref = -12.11      


def calc_H_SI(Tk):                     # H(T) is calculated for every time stamp   and then for the max pressure equation (17), it only 
    dH_T   = np.interp(Tk, _battino_T, _battino_dH)
    dH_avg = ((_dH_ref + dH_T) / 2.0) * 1000.0     
    kH_mol_per_L_per_atm = _kH_ref * np.exp(
        -(dH_avg / R_J) * (1.0 / Tk - 1.0 / _T_ref)
    )
    return kH_mol_per_L_per_atm * 1e3 / 101325.0





def calc_eta(P_peak_kPa, T_peak_K, C_mM, Vsol_ml, Vg_ml):
    P_peak_Pa = P_peak_kPa * 1000.0
    Vg_m3     = ml_to_m3(Vg_ml)
    Vsol_m3   = ml_to_m3(Vsol_ml)
    H_T       = calc_H_SI(T_peak_K)              
    n_gas_max = P_peak_Pa * Vg_m3 / (R_J * T_peak_K)   
    n_aq_max  = H_T * P_peak_Pa * Vsol_m3               
    n_gen_max = mM_ml_to_mol(C_mM, Vsol_ml)             
    return (n_gas_max + n_aq_max) / n_gen_max



def P_model(t, Tk, eta_hat, C_mM, Vsol_ml, Vg_ml, k):
    Vg_m3   = ml_to_m3(Vg_ml)
    Vsol_m3 = ml_to_m3(Vsol_ml)
    H_T     = calc_H_SI(Tk)                             
    n_gen_mol   = mM_ml_to_mol(C_mM, Vsol_ml)           
    numerator   = eta_hat * n_gen_mol * (1.0 - np.exp(-k * t))
    denominator = Vg_m3 / (R_J * Tk) + H_T * Vsol_m3   
    return (numerator / denominator) / 1000.0           



def P_max_model_Hmean(H_mean, eta_hat, C_mM, Vsol_ml, Vg_ml, T_mean):
    Vg_m3   = ml_to_m3(Vg_ml)
    Vsol_m3 = ml_to_m3(Vsol_ml)
    n_gen_mol   = mM_ml_to_mol(C_mM, Vsol_ml)
    denominator = Vg_m3 / (R_J * T_mean) + H_mean * Vsol_m3
    return (eta_hat * n_gen_mol / denominator) / 1000.0  



def interp_piecewise_clamped(x_train, y_train, x_query):
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    order = np.argsort(x)
    return float(np.interp(float(x_query), x[order], y[order]))



def fit_k_models(df_train):
    X = df_train[["C_mM"]].values.astype(float)
    y = df_train["k"].values.astype(float)

    ridge = Ridge(alpha=1e-6)
    ridge.fit(X, y)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))            # Chose this based on the bets choice from the options i this website: https://scikit-learn.org/stable/modules/gaussian_process.html
        * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-1))
    )
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gpr.fit(X, y)

    return ridge, gpr


def predict_k_ridge(ridge, C):
    return float(ridge.predict(np.array([[float(C)]]))[0])


def predict_k_gpr(gpr, C):
    return float(gpr.predict(np.array([[float(C)]]), return_std=False)[0])



experiments = [
    ("M1_RT",  "ABP2", "RT",  1.2,  6.0,  4.0,  "M1_RT_ABP2.csv"),
    ("M2_RT",  "ABP2", "RT",  1.6,  6.0,  4.0,  "M2_RT_ABP2.csv"),
    ("M4_RT",  "ABP2", "RT",  2.5,  6.0,  4.0,  "M4_RT_ABP2.csv"),
   

    ("M1_RT",  "MPR",  "RT",  1.2,  6.0,  4.0,  "M1_RT_MPR.csv"),
   

    ("M1_37C", "ABP2", "37C", 1.2,  6.0,  4.0,  "M1_37C_ABP2_MPR.csv"),
    ("M2_37C", "ABP2", "37C", 1.6,  6.0,  4.0,  "M2_37C_ABP2_MPR.csv"),
    ("M3_37C", "ABP2", "37C", 2.0,  6.0,  4.0,  "M3_37C_ABP2_MPR.csv"),
   

    ("M1_37C", "MPR",  "37C", 1.2,  6.0,  4.0,  "M1_37C_ABP2_MPR.csv"),
    ("M2_37C", "MPR",  "37C", 1.6,  6.0,  4.0,  "M2_37C_ABP2_MPR.csv"),
    ("M3_37C", "MPR",  "37C", 2.0,  6.0,  4.0,  "M3_37C_ABP2_MPR.csv"),

    
    ("M3_45C", "ABP2", "45C", 2.0,  6.0,  4.0,  "M3_45C_ABP2_MPR.csv"),
    ("M3_45C", "MPR",  "45C", 2.0,  6.0,  4.0,  "M3_45C_ABP2_MPR.csv"),
    ("M6_45C", "ABP2", "45C", 2.67, 6.0,  4.0,  "M6_45C_ABP2_MPR.csv"),
    ("M6_45C", "MPR",  "45C", 2.67, 6.0,  4.0,  "M6_45C_ABP2_MPR.csv"),

    
    ("M3_50C", "ABP2", "50C", 2.0,  6.0,  4.0,  "M3_50C_ABP2_MPR.csv"),
    ("M3_50C", "MPR",  "50C", 2.0,  6.0,  4.0,  "M3_50C_ABP2_MPR.csv"),
    ("M6_50C", "ABP2", "50C", 2.67, 6.0,  4.0,  "M6_50C_ABP2_MPR.csv"),
    ("M6_50C", "MPR",  "50C", 2.67, 6.0,  4.0,  "M6_50C_ABP2_MPR.csv"),
]

test_experiments = [
    ("M6_RT",  "ABP2", "RT",  2.67, 5.78, 4.22, "M6_RT_ABP2.csv"),
    ("M6_RT",  "MPR",  "RT",  2.67, 5.78, 4.22, "M6_RT_MPR.csv"),
    ("M6_37C", "ABP2", "37C", 2.67, 5.7,  4.3,  "M6_37C_ABP2_MPR.csv"),
    ("M6_37C", "MPR",  "37C", 2.67, 6.0,  4.0,  "M6_37C_ABP2_MPR.csv"),
    ("M6_45C", "ABP2", "45C", 2.67, 6.0,  4.0,  "M6_45C_ABP2_MPR.csv"),
    ("M6_45C", "MPR",  "45C", 2.67, 6.0,  4.0,  "M6_45C_ABP2_MPR.csv"),
    ("M6_50C", "ABP2", "50C", 2.67, 6.0,  4.0,  "M6_50C_ABP2_MPR.csv"),
    ("M6_50C", "MPR",  "50C", 2.67, 6.0,  4.0,  "M6_50C_ABP2_MPR.csv"),
]


K_VALUES = {
    ("M1_RT",  "ABP2"): 0.0330,
    ("M2_RT",  "ABP2"): 0.0159,
    ("M4_RT",  "ABP2"): 0.0122,


    ("M1_RT",  "MPR") : 0.03930,
    

    ("M1_37C", "ABP2"): 0.09818,
    ("M2_37C", "ABP2"): 0.07972,
    ("M3_37C", "ABP2"): 0.11477,
    

    ("M1_37C", "MPR") : 0.04689,
    ("M2_37C", "MPR") : 0.07874,
    ("M3_37C", "MPR") : 0.04671,

   
    ("M3_45C", "ABP2"): None,
    ("M3_45C", "MPR") : 0.154140,
    ("M6_45C", "ABP2"): None,
    ("M6_45C", "MPR") : 0.156097
,

   
    ("M3_50C", "ABP2"): None,
    ("M3_50C", "MPR") : 0.203281,
    ("M6_50C", "ABP2"): None,
    ("M6_50C", "MPR") : 0.219068,
}



rows = []

for (sample, sensor, temp_label, C_mM, Vsol_ml, Vg_ml, csv_file) in experiments:
    k = K_VALUES.get((sample, sensor))
    if k is None:
        print(f"Skipping {sample} {sensor} — k not yet available")
        continue
    csv_path = os.path.join(CSV_FOLDER, csv_file)
    df = pd.read_csv(csv_path)
    P_col = get_pressure_col(temp_label, sensor)
    t      = df[TIME_COL].values.astype(float);  t -= t[0]
    P_meas = df[P_col].values.astype(float);     P_meas -= P_meas[0]
    Tk = df[TEMP_COL_K].values.astype(float)
    if temp_label != "RT":         # 37C CSVs store Celsius despite column name saying K
        Tk = Tk + 273.15
    idx_peak   = int(np.argmax(P_meas))
    P_peak_kPa = float(P_meas[idx_peak])
    T_peak_K   = float(Tk[idx_peak])

    eta_peak = calc_eta(P_peak_kPa, T_peak_K, C_mM, Vsol_ml, Vg_ml)

    rows.append({
        "sample":        sample,
        "sample_prefix": sample.split("_")[0],
        "sensor":        sensor,
        "temp_label":    temp_label,
        "C_mM":          float(C_mM),
        "Vsol_ml":       float(Vsol_ml),
        "Vg_ml":         float(Vg_ml),
        "k":             float(k),
        "eta_peak":      float(eta_peak),   # η from Eq 13
        "csv":           csv_file,
    })

df_sum = pd.DataFrame(rows)



#
#VALIDATION
for (sample, sensor, temp_label, C_test, Vsol_ml, Vg_ml, csv_test) in test_experiments:

    
    # Use the same eta for both sensors within the same temperature group
    df_eta = df_sum[
        (df_sum["sample_prefix"] != TEST_SAMPLE_PREFIX) &
        (df_sum["temp_label"] == temp_label)
    ].copy()

    # Keep k training sensor-specific
    df_train = df_sum[
        (df_sum["sample_prefix"] != TEST_SAMPLE_PREFIX) &
        (df_sum["sensor"] == sensor) &
        (df_sum["temp_label"] == temp_label)
    ].copy()
    
    if len(df_eta) == 0:
        print(f"Skipping validation for {sample} {sensor} ({temp_label}) — no eta training data available yet")
        continue

    if len(df_train) == 0:
        print(f"Skipping validation for {sample} {sensor} ({temp_label}) — no k training data available yet")
        continue

    eta_hat = float(df_eta["eta_peak"].mean())
    k_interp = interp_piecewise_clamped(df_train["C_mM"], df_train["k"], C_test)
    mean_k = float(df_train["k"].mean())

    
    ridge, gpr = fit_k_models(df_train)
    k_ridge = max(predict_k_ridge(ridge, C_test), 1e-6)
    k_gpr   = max(predict_k_gpr(gpr,   C_test), 1e-6)
    k_interp = max(k_interp, 1e-6)

    
    df_test    = pd.read_csv(os.path.join(CSV_FOLDER, csv_test))
    P_col      = get_pressure_col(temp_label, sensor)
    O2_col     = get_o2_col(sensor)
    t          = df_test[TIME_COL].values.astype(float);  t -= t[0]
    P_meas     = df_test[P_col].values.astype(float);     P_meas -= P_meas[0]
    O2_meas    = df_test[O2_col].values.astype(float);    O2_meas -= O2_meas[0]
    T_nominal  = TEMP_NOMINAL_K[temp_label]
    Tk         = np.full(len(t), T_nominal)

    #P{redicted pressure curves
    P_mean_k  = P_model(t, Tk, eta_hat, C_test, Vsol_ml, Vg_ml, mean_k)
    T_mean    = T_nominal
    H_mean    = float(calc_H_SI(T_nominal))
    Pmax_pred = P_max_model_Hmean(H_mean, eta_hat, C_test, Vsol_ml, Vg_ml, T_mean)

    #predicted O2 curves
    O2_mean_k = O2_model_umol(t, eta_hat, C_test, Vsol_ml, mean_k)

    #metrics
    rmse_P_mean_k,  r2_P_mean_k  = calc_metrics(P_meas,  P_mean_k)
    rmse_O2_mean_k, r2_O2_mean_k = calc_metrics(O2_meas, O2_mean_k)

    print(f"\nTEST SAMPLE : {sample} | Sensor = {sensor} | Temp = {temp_label}")
    print(f"  Pressure  — RMSE = {rmse_P_mean_k:.4f} kPa,  R² = {r2_P_mean_k:.4f}")
    print(f"  O2        — RMSE = {rmse_O2_mean_k:.4f} µmol, R² = {r2_O2_mean_k:.4f}")

    title_base = f"{sample} | {sensor} | {temp_label}  —  Test: {TEST_SAMPLE_PREFIX}"

    # Graph 1 Pressure
    
    # Not using GPR and Ridge predictions for now since they are very close to the mean_k prediction, can be added later for comparison
    plt.figure(figsize=(10, 5))
    plt.plot(t, P_meas,    "k-",  lw=2,   label="Measured ΔP")
    plt.plot(t, P_mean_k,  "b--", lw=1.8, label=f"Predicted (mean k={mean_k:.5f} h⁻¹)  RMSE={rmse_P_mean_k:.3f} kPa  R²={r2_P_mean_k:.3f}")
    plt.xlabel("Time (h)")
    plt.ylabel("ΔP (kPa)")
    plt.title("Pressure — " + title_base)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    # Graph 2  O2 released
    plt.figure(figsize=(10, 5))
    plt.plot(t, O2_meas,   "k-",  lw=2,   label="Measured O₂")
    plt.plot(t, O2_mean_k, "r--", lw=1.8, label=f"Predicted (mean k={mean_k:.5f} h⁻¹)  RMSE={rmse_O2_mean_k:.3f} µmol  R²={r2_O2_mean_k:.3f}")
    plt.xlabel("Time (h)")
    plt.ylabel("O₂ Released (µmol)")
    plt.title("O₂ Released — " + title_base)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


