from pathlib import Path

CSV_FOLDER    = Path(r"C:\Users\chand\Documents\GitHub\Thesis\csv_combi")
OUTPUT_FOLDER = Path("outputs")

TIME_COL      = "Time (h)"
TEMP_COL_K    = "calibrated temperature (K)"

PRESS_COL_RT_ABP2  = "ABP2 DWT denoised pressure (kPa)"
PRESS_COL_RT_MPR   = "MPR SG pressure (kPa)"
PRESS_COL_37C_ABP2 = "ABP2 SG smoothed pressure (kPa)"
PRESS_COL_37C_MPR  = "MPR SG smoothed pressure (kPa)"

R_J = 8.314

TEST_SAMPLE_PREFIX   = "M6"
MAP_SENSOR           = "ABP2"
MAP_CONCENTRATION_MM = 2.67
MAP_TEMPERATURES_C   = [25.0, 37.0]     # temperatures to generate maps for

# Maps to RT label and nominal Tk
TEMP_LABEL_MAP = {25.0: "RT",  37.0: "37C"}
TEMP_TK_MAP    = {25.0: 298.15, 37.0: 310.15}

# Actuation thresholds
PACT_KPA       = 10.0    # minimum pressure to actuate the device (kPa)
P_SAFE_MAX_KPA = 40.0    # maximum safe operating pressure (kPa)
TACT_REQ_H     = 24.0    # maximum allowable actuation time (h) — device must open within this

# Grid definition for (Vsol, Vg) sweep
VSOL_MIN_ML = 0.5
VSOL_MAX_ML = 8.0
VSOL_STEP_ML = 0.5
VG_MIN_ML   = 0.5
VG_MAX_ML   = 8.0
VG_STEP_ML  = 0.5
