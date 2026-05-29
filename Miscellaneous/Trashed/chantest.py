import numpy as np
import matplotlib.pyplot as plt
from math import log, exp
from scipy.constants import R as R_JmolK

# Constants
R = R_JmolK  # Gas constant in J/mol·K
n_total_max = 13.08  # µmol (max theoretical O2 release from 2.18 mM ANT-EPO in 6 ml)
H = 0.0221  # Henry's constant (mol/L/atm), adjust if needed
V_solution = 0.006  # L
V_headspace = 0.002  # L

# Half-lives in seconds from literature
half_life_24C = 35.8 * 3600  # 24°C   Convert from hours to seconds
half_life_37C = 4.7 * 3600   # 37°C

# Converts the half-lives into first-order rate constants to be used
k_24 = log(2) / half_life_24C 
k_37 = log(2) / half_life_37C

# Temperatures in Kelvin
T1 = 297.15  # 24 °C
T2 = 310.15  # 37 °C

# Arrhenius fit from two known rate constants
inv_T1 = 1 / T1
inv_T2 = 1 / T2
ln_k1 = log(k_24)
ln_k2 = log(k_37)

Ea = -R * (ln_k2 - ln_k1) / (inv_T2 - inv_T1)  # Activation energy in J/mol
A = k_24 / exp(-Ea / (R * T1))  # Pre-exponential factor

# Time array (up to 150 hours, in seconds)
time = np.linspace(0, 150 * 3600, 1000)

# Henry’s Law-based model function
def henry_law_model(time, n_total_max, rate_constant, H, V_solution, V_headspace, R_gas=0.08206, T=298.15):
    n_total = n_total_max * (1 - np.exp(-rate_constant * time))  # total O2 produced
    denominator = (V_headspace / (R_gas * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R_gas * T)) / denominator
    n_gas = n_total * partition_fraction  # measurable O2 in headspace
    return n_gas

# Temperatures to simulate (°C)
temps_C = [23, 24, 24.5, 25, 40, 50, 55]
colors = plt.cm.viridis(np.linspace(0, 1, len(temps_C)))

# Plot O₂ released over time for each temperature
plt.figure(figsize=(10, 6))
for i, T_C in enumerate(temps_C):
    T_K = T_C + 273.15
    k_T = A * exp(-Ea / (R * T_K))  # calculate rate constant at this T
    n_gas = henry_law_model(time, n_total_max, k_T, H, V_solution, V_headspace, T=T_K)
    plt.plot(time / 3600, n_gas, label=f'{T_C} °C', color=colors[i])

plt.xlabel('Time (hours)')
plt.ylabel('O₂ Released (µmol)')
plt.title('Simulated O₂ Release at Different Temperatures (2.18 mM ANT-EPO)')
plt.legend(title='Temperature')
plt.grid(True)
plt.tight_layout()
plt.show()
