from pathlib import Path

CSV_FOLDER    = Path(r"C:\Users\chand\Documents\GitHub\Thesis\csv_combi")
OUTPUT_FOLDER = Path("outputs")

TIME_COL      = "Time (h)"
TEMP_COL_K    = "calibrated temperature (K)"
TEMP_COL_37C  = "calibrated temperature (K)"

PRESS_COL_RT_ABP2  = "ABP2 DWT denoised pressure (kPa)"
PRESS_COL_RT_MPR   = "MPR SG pressure (kPa)"
PRESS_COL_37C_ABP2 = "ABP2 SG smoothed pressure (kPa)"
PRESS_COL_37C_MPR  = "MPR SG smoothed pressure (kPa)"

R_J = 8.314

TEST_SAMPLE_PREFIX   = "M6"
MAP_SENSOR           = "ABP2"
MAP_CONCENTRATION_MM = 2.67
MAP_TEMPERATURES_C   = [25.0, 37.0, 50.0]
MAP_TEMP_TO_LABEL    = {25.0: "RT", 37.0: "37C", 50.0: "37C"}

PACT_KPA       = 10.0
P_SAFE_MAX_KPA = 40.0
TACT_SLOW_H    = 30.0
VTOT_ML        = 8.0
VSOL_MIN_ML    = 1.0
VSOL_MAX_ML    = 7.5
VSOL_STEP_ML   = 0.1
VG_MIN_ML      = 0.5

HALF_LIFE_RT_H  = 35.8
HALF_LIFE_37C_H = 4.7
