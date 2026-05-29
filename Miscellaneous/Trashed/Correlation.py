import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import r2_score, mean_squared_error


#read data

def load_pressure_data(file_path):
        
        df = pd.read_csv(file_path)
        clean_df = df.dropna()
        return clean_df
   
    
def henry_law_model(time, n_total_max, rate_constant, H, V_solution=0.006, V_headspace=0.002, R=0.08206, T=298.15):
    """
    Physics-based model using exact equations from measurement protocol:
    
    Henry's Law: C_O2,aq = H * P_O2
    Total partitioning: n_total = n_gas + n_aq  
    Where:
        n_gas = (P * V_headspace) / (R * T)
        n_aq = C_O2,aq * V_solution = H * P * V_solution
    
    Therefore: n_total = P * (V_headspace/(R*T) + H * V_solution)
    Solving for P: P = n_total / (V_headspace/(R*T) + H * V_solution)
    
    But we measure n_gas so:
    substitute P into n_gas, 
    n_gas = P * V_headspace/(R*T) = n_total * (V_headspace/(R*T)) / (V_headspace/(R*T) + H * V_solution)
    
    Parameters:
    - time: Time array (seconds)
    - n_total_max: Maximum total O₂ that can be released (µmol) - corresponds to n_total (in CSV this is called ,Max O2 Possible (µmol))
    - rate_constant: First-order rate constant for ANT-EPO decomposition (1/s)
    - H: Henry's constant (mol/L/atm) - from your protocol
    - V_solution: Solution volume (L) - default 6ml = 0.006L
    - V_headspace: Headspace volume (L) - default 2ml = 0.002L  
    - R: Gas constant (L⋅atm/mol⋅K)
    - T: Temperature (K)
    
    Returns:
    - n_gas: O₂ detectable in headspace via pressure (µmol) - what we measure
    """

    # Step 1: Total O₂ released from ANT-EPO decomposition (first-order kinetics). (to find the practical max o2. Theoretical max is in the CSV= 13.08)
    n_total = n_total_max * (1 - np.exp(-rate_constant * time))
    
    # Step 2
    
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    
    # Step 3: Detectable O₂ in headspace (what pressure sensor measures )
    n_gas = n_total * partition_fraction
    
    return n_gas

def fit_henry_model(df, time_col='Time (s)', o2_col='O2 Released (µmol)', max_o2_col='Max O2 Possible (µmol)'):
    """
    Fit Henry's Law model to experimental pressure data using protocol equations
    """
    
    # Extract data
    time_data = df[time_col].values
    n_gas_measured = df[o2_col].values  # This is what pressure sensor detects               PHYSICAL
    n_total_theoretical = df[max_o2_col].iloc[0]  # Maximum possible from stoichiometry      THEORETICAL
    
    # Get max measured value first
    max_measured = n_gas_measured.max() #in m2 it is 4.8
    
    # n_total_max: Must be larger than max measured, but reasonable, Since only ~37% is detected, actual n_total is probably close to theoretical
    estimated_n_total_max = max_measured / 0.4  # Assume ~40% detection efficiency
    estimated_n_total_max = min(estimated_n_total_max, n_total_theoretical)
    
    # rate_constant: Estimate from data (1/time to reach 63% of max)
    time_63_percent = time_data[np.argmin(np.abs(n_gas_measured - 0.63 * max_measured))]
    initial_rate = 1.0 / time_63_percent if time_63_percent > 0 else 1e-5

    """""
    V_headspace = 0.002 L
    V_solution = 0.006 L
    R = 0.08206 L⋅atm/mol⋅K
    T = 310.15 K (37°C)
    n_gas (measured max) = 4.853248 µmol = 4.853248 × 10⁻⁶ mol
    n_total (theoretical) = 13.08 µmol = 13.08 × 10⁻⁶ mol

    Step 1: Calculate pressure P
    From: n_gas = P × V_headspace/(R×T)
    Therefore: P = n_gas × (R×T) / V_headspace
    P = (4.853248 × 10⁻⁶) × (0.08206) × (310.15) / (0.002)
    P = 0.0619 atm
    Step 2: Calculate H using your protocol equation
    From: n_total = P × (V_headspace/(R×T) + H × V_solution)
    Rearranging: H = (n_total/P - V_headspace/(R×T)) / V_solution
    V_headspace/(R×T) = 0.002/(0.08206 × 310.15) = 7.858 × 10⁻⁵ mol/atm
    H = ((13.08 × 10⁻⁶)/0.0619 - 7.858 × 10⁻⁵) / 0.006
    H = (2.113 × 10⁻⁴ - 7.858 × 10⁻⁵) / 0.006
    H = 1.327 × 10⁻⁴ / 0.006
    H = 2.21 × 10⁻² mol/L/atm
    """

    initial_H = 0.0221
    initial_params = [estimated_n_total_max, initial_rate, initial_H]

    
    # Parameter bounds (physically reasonable ranges)
    # Fix bounds based on actual data range vs theoretical
    
    # n_total_max should be between max_measured and theoretical (but closer to theoretical)
    lower_bounds = [max_measured * 1.1, 1e-7, 1e-5]  # Must be larger than max measured ( ~ 12 , close to therotical value, this is the o2 that really wasnt released but rather dissolved)
    upper_bounds = [n_total_theoretical * 3, 1e-1, 1e-1]  # Increased rate constant upper bound
    
    # Make sure initial guesses are within bounds
    initial_params[0] = max(initial_params[0], lower_bounds[0])
    initial_params[0] = min(initial_params[0], upper_bounds[0])
    initial_params[1] = max(initial_params[1], lower_bounds[1])
    initial_params[1] = min(initial_params[1], upper_bounds[1])
    initial_params[2] = max(initial_params[2], lower_bounds[2])
    initial_params[2] = min(initial_params[2], upper_bounds[2])
    
    # print(f"Parameter bounds:")
    # print(f"n_total_max: {lower_bounds[0]:.3f} to {upper_bounds[0]:.3f} µmol")
    # print(f"rate_constant: {lower_bounds[1]:.2e} to {upper_bounds[1]:.2e} s⁻¹") 
    # print(f"H: {lower_bounds[2]:.2e} to {upper_bounds[2]:.2e} mol/L/atm")
    
    # print(f"\nAdjusted initial parameters (within bounds):")
    # print(f"n_total_max: {initial_params[0]:.3f} µmol")
    # print(f"rate_constant: {initial_params[1]:.2e} s⁻¹")
    # print(f"H (Henry's constant): {initial_params[2]:.2e} mol/L/atm")
    
    try:
        # Fit the model to the data   #using NLS
        popt, pcov = curve_fit(
            henry_law_model, time_data, n_gas_measured,p0=initial_params,bounds=(lower_bounds, upper_bounds),maxfev=5000
        )
        
        # Extract fitted parameters
        fitted_n_total_max, fitted_rate, fitted_H = popt
        param_errors = np.sqrt(np.diag(pcov))
        
        
        # Calculate derived quantities using protocol equations
        V_headspace = 0.002  # L
        V_solution = 0.006   # L  
        R = 0.08206         # L⋅atm/mol⋅K
        T = 298.15          # K
        
        denominator = (V_headspace / (R * T)) + (fitted_H * V_solution)
        partition_fraction = (V_headspace / (R * T)) / denominator
    
        
        # Calculate fit quality
        n_gas_predicted = henry_law_model(time_data, *popt)
        r2 = r2_score(n_gas_measured, n_gas_predicted)
        rmse = np.sqrt(mean_squared_error(n_gas_measured, n_gas_predicted))
        
        print(f"\nMODEL FIT QUALITY:")
        print(f"R² score:         {r2:.4f}")
        print(f"RMSE:            {rmse:.6f} µmol")
        
        return popt, pcov, r2, rmse, time_data, n_gas_measured, n_gas_predicted
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None, None, None, None, None, None, None
    


def predict_new_ratio(fitted_params, new_V_solution, new_V_headspace, time_points):
    """
    Predict n_gas (detectable O₂) for different solution/headspace volume ratios
    """
    
    fitted_n_total_max, fitted_rate, fitted_H = fitted_params  #popt is renamed as fitted_params for clarity.
    
    # Scale n_total_max for new solution volume (more ANT-EPO in larger volume)
    # Assuming same concentration: new_n_total_max = old_n_total_max * (new_vol/old_vol)
    old_V_solution = 0.006  # 6ml
    scaled_n_total_max = fitted_n_total_max * (new_V_solution / old_V_solution)
    
    # Predict with new volumes using protocol equations
    predicted_n_gas = henry_law_model(time_points, scaled_n_total_max, fitted_rate,fitted_H, V_solution=new_V_solution, V_headspace=new_V_headspace)

    return predicted_n_gas, scaled_n_total_max   #predicted_n_gas: How much will be detected with new ratio




def analyze_ratio_effects(fitted_params, max_time=100000):
    """
    Analyze how different solution/headspace ratios affect n_gas (detectable O₂)
    """
   
    
    # Define different volume combinations
    scenarios = [
        {"name": "Current (6ml:2ml)", "V_sol": 0.006, "V_head": 0.002, "ratio": 3.0},
        {"name": "Your target (8ml:4ml)", "V_sol": 0.008, "V_head": 0.004, "ratio": 2.0},
        {"name": "Optimized (4ml:4ml)", "V_sol": 0.004, "V_head": 0.004, "ratio": 1.0},
        {"name": "High headspace (6ml:6ml)", "V_sol": 0.006, "V_head": 0.006, "ratio": 1.0},
        {"name": "Low ratio (10ml:5ml)", "V_sol": 0.010, "V_head": 0.005, "ratio": 2.0}
    ]
    
    time_final = np.array([max_time])  # Final steady-state
    results = []
    
    print(f"{'Scenario':<25} {'Ratio':<8} {'n_gas (µmol)':<15} {'n_total (µmol)':<15} {'Efficiency (%)':<15}")
    print("-" * 85)
    
    baseline_n_gas = None
    #for each scenario, call predict_new_ratio to get prediction for that volume combination
    for scenario in scenarios:
        predicted_n_gas, scaled_n_total_max = predict_new_ratio(fitted_params, scenario["V_sol"], scenario["V_head"], time_final)
        
        final_n_gas = predicted_n_gas[0]
        efficiency = (final_n_gas / scaled_n_total_max) * 100   #efficiency = (detected O₂ / total O₂) × 100%
        
        if baseline_n_gas is None and "Current" in scenario["name"]:
            baseline_n_gas = final_n_gas
        
        improvement = ((final_n_gas - baseline_n_gas) / baseline_n_gas * 100) if baseline_n_gas else 0
        
        print(f"{scenario['name']:<25} {scenario['ratio']:<8.1f} {final_n_gas:<15.6f} {scaled_n_total_max:<15.3f} {efficiency:<15.1f}")
        
        results.append({
            'scenario': scenario['name'],
            'ratio': scenario['ratio'],
            'V_solution': scenario['V_sol'],
            'V_headspace': scenario['V_head'],
            'final_n_gas': final_n_gas,
            'scaled_n_total_max': scaled_n_total_max,
            'efficiency': efficiency,
            'improvement': improvement
        })
    
    return results

def plot_henry_model_results(time_data, n_gas_measured, n_gas_predicted, fitted_params, r2, rmse):
    """
    Plot model fit and predictions for different ratios - GENERAL ANALYSIS
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Model fit vs experimental data
    if len(time_data) > 5000:
        indices = np.random.choice(len(time_data), 5000, replace=False)
        time_plot = time_data[indices]
        measured_plot = n_gas_measured[indices]
        predicted_plot = n_gas_predicted[indices]
    else:
        time_plot = time_data
        measured_plot = n_gas_measured
        predicted_plot = n_gas_predicted
    
    ax1.scatter(time_plot/3600, measured_plot, alpha=0.6, s=10, label='Experimental n_gas', color='blue')
    ax1.plot(time_plot/3600, predicted_plot, 'r-', linewidth=2, label='Henry Model Fit')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('n_gas (µmol) - Detectable O₂')
    ax1.set_title(f'Model Fit (R² = {r2:.3f}, RMSE = {rmse:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = n_gas_measured - n_gas_predicted
    ax2.scatter(n_gas_predicted, residuals, alpha=0.6, s=10)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted n_gas (µmol)')
    ax2.set_ylabel('Residuals (µmol)')
    ax2.set_title('Model Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Multiple ratio scenario predictions (GENERAL - not focused on 2.0)
    time_pred = np.linspace(0, time_data.max(), 1000)
    ratio_scenarios = [
        {"name": "Current (3.0)", "V_sol": 0.006, "V_head": 0.002, "color": "blue", "style": "-"},
        {"name": "Lower ratio (1.5)", "V_sol": 0.006, "V_head": 0.004, "color": "green", "style": "-"},
        {"name": "Equal volumes (1.0)", "V_sol": 0.004, "V_head": 0.004, "color": "orange", "style": "-"},
        {"name": "Higher headspace (0.75)", "V_sol": 0.003, "V_head": 0.004, "color": "purple", "style": "--"}
    ]
    
    for scenario in ratio_scenarios:
        pred_n_gas, _ = predict_new_ratio(
            fitted_params, 
            scenario["V_sol"], 
            scenario["V_head"], 
            time_pred
        )
        ax3.plot(time_pred/3600, pred_n_gas, linewidth=2, linestyle=scenario["style"],
                label=scenario["name"], color=scenario["color"])
    
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Predicted n_gas (µmol)')
    ax3.set_title('Ratio Comparison: General Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Detection efficiency vs ratio (CONTINUOUS ANALYSIS)
    ratios = np.linspace(0.5, 5, 100)  # More points for smoother curve
    efficiencies = []
    
    R = 0.08206  # L⋅atm/mol⋅K
    T = 298.15   # K
    fitted_H = fitted_params[2]
    
    for ratio in ratios:
        # Keep total volume constant, vary ratio
        total_vol = 0.008  # 8ml total
        V_head = total_vol / (1 + ratio)
        V_sol = total_vol - V_head
        
        # Using protocol equation
        denominator = (V_head / (R * T)) + (fitted_H * V_sol)
        partition_fraction = (V_head / (R * T)) / denominator
        efficiency = partition_fraction * 100
        
        efficiencies.append(efficiency)
    
    ax4.plot(ratios, efficiencies, 'b-', linewidth=2, label='Detection Efficiency')
    
    # Mark current ratio and show general optimal range
    ax4.axvline(x=3.0, color='blue', linestyle='--', alpha=0.7, label='Current (3.0)')
    
    # Highlight optimal range instead of specific values
    optimal_ratios = ratios[np.array(efficiencies) > np.max(efficiencies) * 0.9]  # Within 90% of max
    if len(optimal_ratios) > 0:
        ax4.axvspan(optimal_ratios.min(), optimal_ratios.max(), 
                   alpha=0.2, color='green', label='Optimal Range (>90% max efficiency)')
    
    ax4.set_xlabel('Solution/Headspace Ratio (V_solution/V_headspace)')
    ax4.set_ylabel('Detection Efficiency: n_gas/n_total (%)')
    ax4.set_title('Detection Efficiency vs. Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_ratio_effects(fitted_params, max_time=100000):
    """
    Analyze how different solution/headspace ratios affect n_gas (detectable O₂)
    GENERAL ANALYSIS - multiple scenarios
    """
    
    # Define DIVERSE volume combinations (not focused on any specific target)
    scenarios = [
        {"name": "Current setup (6ml:2ml)", "V_sol": 0.006, "V_head": 0.002, "ratio": 3.0},
        {"name": "Lower ratio (6ml:4ml)", "V_sol": 0.006, "V_head": 0.004, "ratio": 1.5},
        {"name": "Equal volumes (4ml:4ml)", "V_sol": 0.004, "V_head": 0.004, "ratio": 1.0},
        {"name": "More headspace (3ml:4ml)", "V_sol": 0.003, "V_head": 0.004, "ratio": 0.75},
        {"name": "Larger system (8ml:4ml)", "V_sol": 0.008, "V_head": 0.004, "ratio": 2.0},
        {"name": "High headspace (6ml:6ml)", "V_sol": 0.006, "V_head": 0.006, "ratio": 1.0}
    ]
    
    time_final = np.array([max_time])  # Final steady-state
    results = []
    
    print(f"\n" + "="*80)
    print("GENERAL RATIO EFFECT ANALYSIS")
    print("="*80)
    print(f"{'Scenario':<28} {'Ratio':<8} {'n_gas (µmol)':<15} {'n_total (µmol)':<15} {'Efficiency (%)':<15}")
    print("-" * 88)
    
    baseline_n_gas = None
    
    for scenario in scenarios:
        predicted_n_gas, scaled_n_total_max = predict_new_ratio(
            fitted_params, scenario["V_sol"], scenario["V_head"], time_final
        )
        
        final_n_gas = predicted_n_gas[0]
        efficiency = (final_n_gas / scaled_n_total_max) * 100
        
        if baseline_n_gas is None and "Current" in scenario["name"]:
            baseline_n_gas = final_n_gas
        
        improvement = ((final_n_gas - baseline_n_gas) / baseline_n_gas * 100) if baseline_n_gas else 0
        
        print(f"{scenario['name']:<28} {scenario['ratio']:<8.2f} {final_n_gas:<15.6f} {scaled_n_total_max:<15.3f} {efficiency:<15.1f}")
        
        results.append({
            'scenario': scenario['name'],
            'ratio': scenario['ratio'],
            'V_solution': scenario['V_sol'],
            'V_headspace': scenario['V_head'],
            'final_n_gas': final_n_gas,
            'scaled_n_total_max': scaled_n_total_max,
            'efficiency': efficiency,
            'improvement': improvement
        })
    
    # Add summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Find best and worst scenarios
    best_scenario = max(results, key=lambda x: x['final_n_gas'])
    worst_scenario = min(results, key=lambda x: x['final_n_gas'])
    
    print(f"🏆 Best detection: {best_scenario['scenario']} ({best_scenario['final_n_gas']:.3f} µmol)")
    print(f"📉 Worst detection: {worst_scenario['scenario']} ({worst_scenario['final_n_gas']:.3f} µmol)")
    
    # General trend
    print(f"📊 General trend: Lower ratios tend to improve oxygen detection")
    print(f"🔬 Physics: More headspace volume → better gas-phase partitioning")
    
    return results

def answer_general_ratio_question(fitted_params):
    """
    Provide general guidance on ratio optimization (not focused on specific target)
    """
    
    print(f"\n" + "="*60)
    print("GENERAL RATIO OPTIMIZATION GUIDANCE")
    print("="*60)
    
    # Current setup
    current_n_gas, current_n_total_max = predict_new_ratio(
        fitted_params, 0.006, 0.002, np.array([100000])
    )
    current_detectable = current_n_gas[0]
    
    # Test multiple ratios to find optimal range
    test_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    ratio_results = []
    
    print(f"Current setup (6ml:2ml, ratio=3.0): {current_detectable:.6f} µmol detected")
    print(f"\nTesting different ratios:")
    print(f"{'Ratio':<8} {'Detection (µmol)':<18} {'Improvement (%)':<18}")
    print("-" * 50)
    
    for ratio in test_ratios:
        # Keep total volume constant at 8ml
        total_vol = 0.008
        V_head = total_vol / (1 + ratio)
        V_sol = total_vol - V_head
        
        test_n_gas, _ = predict_new_ratio(fitted_params, V_sol, V_head, np.array([100000]))
        test_detectable = test_n_gas[0]
        improvement = ((test_detectable - current_detectable) / current_detectable) * 100
        
        ratio_results.append({'ratio': ratio, 'detection': test_detectable, 'improvement': improvement})
        print(f"{ratio:<8.2f} {test_detectable:<18.6f} {improvement:<18.1f}")
    
    # Find optimal ratio
    best_ratio_result = max(ratio_results, key=lambda x: x['detection'])
    
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"🏆 Optimal ratio: ~{best_ratio_result['ratio']:.1f}")
    print(f"📈 Maximum detection: {best_ratio_result['detection']:.6f} µmol")
    print(f"🚀 Potential improvement: {best_ratio_result['improvement']:.1f}%")
    
    print(f"\n💡 GENERAL RECOMMENDATIONS:")
    
    # Categorize improvement levels
    significant_improvements = [r for r in ratio_results if r['improvement'] > 20]
    moderate_improvements = [r for r in ratio_results if 10 <= r['improvement'] <= 20]
    
    if significant_improvements:
        ratios_list = [f"{r['ratio']:.1f}" for r in significant_improvements]
        print(f"✅ Significant improvement (>20%): Ratios {', '.join(ratios_list)}")
    
    if moderate_improvements:
        ratios_list = [f"{r['ratio']:.1f}" for r in moderate_improvements]
        print(f"🔶 Moderate improvement (10-20%): Ratios {', '.join(ratios_list)}")
    
    print(f"🔬 Physics principle: Lower ratios = more headspace = better O₂ detection")
    
    return ratio_results

# Update main function
def main_henry_analysis(file_path):
    """
    Complete Henry's Law analysis pipeline - GENERAL RATIO ANALYSIS
    """
    print("PHYSICS-BASED OXYGEN PARTITIONING ANALYSIS")
    print("General Ratio Optimization Study")
    print("=" * 60)
    
    # Load data
    df = load_pressure_data(file_path)
    if df is None:
        return
    
    # Fit Henry model
    fit_results = fit_henry_model(df)
    if fit_results[0] is None:
        return
    
    popt, pcov, r2, rmse, time_data, n_gas_measured, n_gas_predicted = fit_results
    
    # Analyze ratio effects (general)
    ratio_results = analyze_ratio_effects(popt)
    
    # General optimization guidance
    optimization_results = answer_general_ratio_question(popt)
    
    # Create visualizations (general)
    plot_henry_model_results(time_data, n_gas_measured, n_gas_predicted, popt, r2, rmse)
    
    print(f"\n" + "="*60)
    print("GENERAL ANALYSIS COMPLETE!")
    print("="*60)
    print("Use these results to choose the optimal ratio for your specific needs!")

# Usage remains the same
if __name__ == "__main__":
    file_path = "M2_2.18mM_pressure_o2_release.csv"
    main_henry_analysis(file_path)
