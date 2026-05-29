from pathlib import Path

# =========================
# PATHS
# =========================
CSV_FOLDER = Path(r"C:\Users\chand\Documents\GitHub\Thesis\csv_combi")
OUTPUT_FOLDER = Path("outputs")

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

# Temperatures requested by supervisor
MAP_TEMPERATURES_C = [25.0, 37.0, 50.0]

# For mapping temperature -> training label
# 25 C uses RT-trained parameters
# 37 C uses 37C-trained parameters
# 50 C currently also uses 37C-trained parameters unless you later add 50 C training data
MAP_TEMP_TO_LABEL = {
    25.0: "RT",
    37.0: "37C",
    50.0: "37C",
}

# Geometry sweep ranges
VSOL_MIN_ML = 1.0
VSOL_MAX_ML = 8.0
VSOL_STEP_ML = 0.1

VG_MIN_ML = 0.5
VG_MAX_ML = 8.0
VG_STEP_ML = 0.1

# Actuator threshold
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