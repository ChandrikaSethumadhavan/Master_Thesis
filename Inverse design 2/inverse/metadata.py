# All available experiments

EXPERIMENTS = [
    ("M1_RT",  "ABP2", "RT",  1.2,  6.0,  4.0,  "M1_RT_ABP2.csv"),
    ("M2_RT",  "ABP2", "RT",  1.6,  6.0,  4.0,  "M2_RT_ABP2.csv"),
    ("M4_RT",  "ABP2", "RT",  2.5,  6.0,  4.0,  "M4_RT_ABP2.csv"),
    ("M6_RT",  "ABP2", "RT",  2.67, 5.78, 4.22, "M6_RT_ABP2.csv"),

    ("M1_RT",  "MPR",  "RT",  1.2,  6.0,  4.0,  "M1_RT_MPR.csv"),
    ("M6_RT",  "MPR",  "RT",  2.67, 5.78, 4.22, "M6_RT_MPR.csv"),

    ("M1_37C", "ABP2", "37C", 1.2,  6.0,  4.0,  "M1_37C_ABP2_MPR.csv"),
    ("M2_37C", "ABP2", "37C", 1.6,  6.0,  4.0,  "M2_37C_ABP2_MPR.csv"),
    ("M3_37C", "ABP2", "37C", 2.0,  6.0,  4.0,  "M3_37C_ABP2_MPR.csv"),
    ("M6_37C", "ABP2", "37C", 2.67, 5.7,  4.3,  "M6_37C_ABP2_MPR.csv"),

    ("M1_37C", "MPR",  "37C", 1.2,  6.0,  4.0,  "M1_37C_ABP2_MPR.csv"),
    ("M2_37C", "MPR",  "37C", 1.6,  6.0,  4.0,  "M2_37C_ABP2_MPR.csv"),
    ("M3_37C", "MPR",  "37C", 2.0,  6.0,  4.0,  "M3_37C_ABP2_MPR.csv"),
    ("M6_37C", "MPR",  "37C", 2.67, 6.0,  4.0,  "M6_37C_ABP2_MPR.csv"),
]

# k values per (sample, sensor)
K_VALUES = {
    ("M1_RT",  "ABP2"): 0.0330,
    ("M2_RT",  "ABP2"): 0.0159,
    ("M4_RT",  "ABP2"): 0.0122,
    ("M6_RT",  "ABP2"): 0.0087,

    ("M1_RT",  "MPR") : 0.03930,
    ("M6_RT",  "MPR") : 0.006921,

    ("M1_37C", "ABP2"): 0.09818,
    ("M2_37C", "ABP2"): 0.07972,
    ("M3_37C", "ABP2"): 0.11477,
    ("M6_37C", "ABP2"): 0.118947,

    ("M1_37C", "MPR") : 0.04689,
    ("M2_37C", "MPR") : 0.07874,
    ("M3_37C", "MPR") : 0.04671,
    ("M6_37C", "MPR") : 0.079144,
}