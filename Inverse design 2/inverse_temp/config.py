"""
Config for temperature-aware Pareto optimisation.
Extends the base inverse/config.py with a temperature sweep.
All other settings (paths, sensor, pressure limits) are inherited.
"""
import sys
import os

from pathlib import Path

# =========================
# PATHS
# =========================
CSV_FOLDER = Path(r"C:\Users\chand\Documents\GitHub\Thesis\csv_combi")


# =========================
# USER SETTINGS
# =========================
TIME_COL = "Time (h)"
TEMP_COL_K = "calibrated temperature (K)"
TEMP_COL_37C = "calibrated temperature (K)"

PRESS_COL_RT_ABP2  = "ABP2 DWT denoised pressure (kPa)"
PRESS_COL_RT_MPR   = "MPR SG pressure (kPa)"
PRESS_COL_37C_ABP2 = "ABP2 SG smoothed pressure (kPa)"
PRESS_COL_37C_MPR  = "MPR SG smoothed pressure (kPa)"

R_J = 8.314
R_GAS = 8.314
R_ATM = 0.082057

# Which family to exclude from training
TEST_SAMPLE_PREFIX = "M6"

# Which sensor to use for design maps
MAP_SENSOR = "ABP2"

# Fixed concentration for initial supervisor-style maps
MAP_CONCENTRATION_MM = 2.0
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inverse'))

# ── inherit everything from the base config ──────────────────────────────────
from config import *   # noqa: F401, F403

# ── override output folder so plots go into inverse_temp/outputs ─────────────
from pathlib import Path
OUTPUT_FOLDER = Path(os.path.join(os.path.dirname(__file__), "outputs"))

# ── temperature sweep for the Pareto search ──────────────────────────────────
# Range: RT (25 °C) up to body temperature (37 °C) in 1 °C steps.
# Extend T_MAX_C to 50 if you want to explore fever-range conditions later.
T_MIN_C  = 25.0
T_MAX_C  = 37.0
T_STEP_C = 1.0     # 1 °C resolution — fine enough for Arrhenius interpolation

T_REF_RT_K  = 298.15   # reference for Arrhenius scaling (RT group)
T_REF_37C_K = 310.15   # reference for 37 °C group

# ── independent Vg sweep (not linked to Vtot) ────────────────────────────────
VG_MAX_ML  = 7.5
VG_STEP_ML = 0.5   # coarser than Vsol to keep sweep fast

#Actuator threshold
PACT_KPA = 10.0

# Time settings
TIME_MAX_H = 100.0
TIME_STEP_H = 0.1

# Safety upper pressure limit for 3-zone maps
P_SAFE_MAX_KPA = 40.0

# "Too slow" threshold for combined map
TACT_SLOW_H = 30.0

# Number of recommended designs to show
N_TOP_RECOMMENDATIONS = 5

# Sensitivity settings
SENSITIVITY_BASE_T_C = 37.0
SENSITIVITY_BASE_C_MM = 2.0
SENSITIVITY_BASE_VSOL_ML = 6.0
SENSITIVITY_BASE_VG_ML = 2.0
SENSITIVITY_DELTA_FRAC = 0.20




VTOT_ML = 8.0
P_SAFE_MAX_KPA = 40.0
N_TOP_RECOMMENDATIONS = 5

SENSITIVITY_BASE_T_C = 37.0
SENSITIVITY_BASE_C_MM = 2.0
SENSITIVITY_BASE_VSOL_ML = 6.0
SENSITIVITY_BASE_VG_ML = 2.0
SENSITIVITY_DELTA_FRAC = 0.20



VSOL_MIN_ML = 1.0
VSOL_MAX_ML = 7.5   # VTOT - practical min headspace (0.5 mL)
VSOL_STEP_ML = 0.1

VG_MIN_ML = 0.5     # minimum allowed headspace

# Known biological half-lives of ANT-EPO from experiments
# Used to draw a physically-motivated minimum t_act line on the actuation map.
# Minimum t_act = -ln(1 - Pact/Psafe) / k_bio  where k_bio = ln(2) / t_half
HALF_LIFE_RT_H  = 35.8   # hours at room temperature (25 °C)
HALF_LIFE_37C_H = 4.7    # hours at 37 °C

# Vtot sweep for Pareto multi-objective search
# C is fixed at MAP_CONCENTRATION_MM; only Vsol and Vtot vary.
VTOT_MIN_ML  = 2.0
VTOT_MAX_ML  = 12.0
VTOT_STEP_ML = 0.5