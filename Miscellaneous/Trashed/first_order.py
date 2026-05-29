#interactive pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import r2_score, mean_squared_error

# ... (keep all previous functions: load_pressure_data, henry_law_model, fit_henry_model, predict_new_ratio) ...


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



def interactive_ratio_explorer(fitted_params):
    """
    Interactive pipeline for exploring different volume ratios
    """
    
    print("\n" + "="*70)
    print("🔬 INTERACTIVE OXYGEN DETECTION PREDICTOR")
    print("="*70)
    print("Enter your desired volume combinations to predict oxygen detection!")
    print("Type 'quit' to exit, 'help' for guidance, or 'summary' for analysis overview")
    
    # Store user predictions for comparison
    user_predictions = []
    
    # Get current baseline for reference
    current_n_gas, current_n_total_max = predict_new_ratio(
        fitted_params, 0.006, 0.002, np.array([100000])
    )
    baseline_detection = current_n_gas[0]
    
    print(f"\n📊 REFERENCE: Current setup (6ml:2ml) detects {baseline_detection:.6f} µmol")
    
    while True:
        print("\n" + "-"*50)
        user_input = input("Enter command or volumes: ").strip().lower()
        
        # Handle special commands
        if user_input == 'quit':
            break
        elif user_input == 'help':
            show_help()
            continue
        elif user_input == 'summary':
            show_prediction_summary(user_predictions, baseline_detection)
            continue
        elif user_input == 'compare':
            compare_scenarios(fitted_params, user_predictions, baseline_detection)
            continue
        elif user_input == 'optimize':
            find_optimal_ratio(fitted_params, baseline_detection)
            continue
        
        # Parse volume input
        try:
            prediction_result = process_volume_input(user_input, fitted_params, baseline_detection)
            if prediction_result:
                user_predictions.append(prediction_result)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try format: '8ml:4ml' or 'ratio 2.0' or type 'help'")
        
    # Final summary
    if user_predictions:
        print(f"\n" + "="*70)
        print("📋 SESSION SUMMARY")
        print("="*70)
        show_prediction_summary(user_predictions, baseline_detection)
    
    print("\n👋 Thanks for using the Interactive Oxygen Detection Predictor!")

def process_volume_input(user_input, fitted_params, baseline_detection):
    """
    Process different types of volume input formats
    """
    
    # Format 1: "8ml:4ml" or "8:4"
    if ':' in user_input:
        parts = user_input.replace('ml', '').split(':')
        if len(parts) == 2:
            V_sol = float(parts[0]) / 1000  # Convert ml to L
            V_head = float(parts[1]) / 1000
            ratio = V_sol / V_head
            return make_prediction(V_sol, V_head, ratio, fitted_params, baseline_detection, user_input)
    
    # Format 2: "ratio 2.0" or "r 2.0"
    elif user_input.startswith(('ratio ', 'r ')):
        ratio_str = user_input.split()[-1]
        ratio = float(ratio_str)
        
        # Ask for total volume
        total_vol_input = input(f"Total volume for ratio {ratio} (default 8ml): ").strip()
        if total_vol_input == '':
            total_vol = 0.008  # 8ml default
        else:
            total_vol = float(total_vol_input.replace('ml', '')) / 1000
        
        V_head = total_vol / (1 + ratio)
        V_sol = total_vol - V_head
        
        return make_prediction(V_sol, V_head, ratio, fitted_params, baseline_detection, 
                             f"ratio {ratio} ({V_sol*1000:.1f}ml:{V_head*1000:.1f}ml)")
    
    # Format 3: Just numbers "8 4"
    elif len(user_input.split()) == 2:
        parts = user_input.split()
        V_sol = float(parts[0]) / 1000
        V_head = float(parts[1]) / 1000
        ratio = V_sol / V_head
        return make_prediction(V_sol, V_head, ratio, fitted_params, baseline_detection, 
                             f"{parts[0]}ml:{parts[1]}ml")
    
    else:
        raise ValueError("Invalid format")

def make_prediction(V_sol, V_head, ratio, fitted_params, baseline_detection, input_description):
    """
    Make prediction and display results
    """
    
    # Validate inputs
    if V_sol <= 0 or V_head <= 0:
        raise ValueError("Volumes must be positive")
    if V_sol > 0.020 or V_head > 0.020:  # 20ml limit
        print("⚠️  Warning: Very large volumes may be impractical")
    
    # Make prediction
    pred_n_gas, scaled_n_total_max = predict_new_ratio(
        fitted_params, V_sol, V_head, np.array([100000])
    )
    
    predicted_detection = pred_n_gas[0]
    efficiency = (predicted_detection / scaled_n_total_max) * 100
    improvement = ((predicted_detection - baseline_detection) / baseline_detection) * 100
    
    # Display results
    print(f"\n🔬 PREDICTION RESULTS for {input_description}:")
    print(f"   Ratio: {ratio:.2f}")
    print(f"   Predicted detection: {predicted_detection:.6f} µmol")
    print(f"   Detection efficiency: {efficiency:.1f}%")
    print(f"   vs. Current setup: {improvement:+.1f}% change")
    
    # Give qualitative assessment
    if improvement > 20:
        print(f"   ✅ EXCELLENT: Significant improvement!")
    elif improvement > 10:
        print(f"   🔶 GOOD: Moderate improvement")
    elif improvement > -10:
        print(f"   ➡️  SIMILAR: Minimal change")
    else:
        print(f"   ❌ WORSE: Lower detection than current")
    
    return {
        'input': input_description,
        'V_sol': V_sol,
        'V_head': V_head,
        'ratio': ratio,
        'detection': predicted_detection,
        'efficiency': efficiency,
        'improvement': improvement
    }

def show_help():
    """
    Display help information
    """
    print(f"\n💡 INPUT FORMATS:")
    print(f"   📝 '8ml:4ml' or '8:4' - Direct volume specification")
    print(f"   📝 'ratio 2.0' or 'r 2.0' - Specify ratio (will ask for total volume)")
    print(f"   📝 '8 4' - Two numbers (solution ml, headspace ml)")
    
    print(f"\n🔧 COMMANDS:")
    print(f"   📝 'help' - Show this help")
    print(f"   📝 'summary' - Show all your predictions")
    print(f"   📝 'compare' - Compare your predictions visually")
    print(f"   📝 'optimize' - Find optimal ratio automatically")
    print(f"   📝 'quit' - Exit the predictor")
    
    print(f"\n💡 EXAMPLES:")
    print(f"   📝 '10ml:5ml' → Tests 10ml solution, 5ml headspace")
    print(f"   📝 'ratio 1.5' → Tests ratio 1.5 with 8ml total (default)")
    print(f"   📝 '6 6' → Tests 6ml solution, 6ml headspace")

def show_prediction_summary(predictions, baseline):
    """
    Show summary of all user predictions
    """
    if not predictions:
        print("📭 No predictions made yet!")
        return
    
    print(f"\n📊 YOUR PREDICTIONS SUMMARY:")
    print(f"{'Setup':<20} {'Ratio':<8} {'Detection (µmol)':<18} {'Improvement':<12}")
    print("-" * 65)
    
    # Sort by improvement
    sorted_predictions = sorted(predictions, key=lambda x: x['improvement'], reverse=True)
    
    for pred in sorted_predictions:
        improvement_str = f"{pred['improvement']:+.1f}%"
        print(f"{pred['input']:<20} {pred['ratio']:<8.2f} {pred['detection']:<18.6f} {improvement_str:<12}")
    
    # Highlight best and worst
    best = sorted_predictions[0]
    worst = sorted_predictions[-1]
    
    print(f"\n🏆 BEST: {best['input']} ({best['improvement']:+.1f}% improvement)")
    print(f"📉 WORST: {worst['input']} ({worst['improvement']:+.1f}% change)")

def compare_scenarios(fitted_params, user_predictions, baseline):
    """
    Create visual comparison of user predictions
    """
    if len(user_predictions) < 2:
        print("📊 Need at least 2 predictions to compare!")
        return
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Detection comparison
    plt.subplot(2, 2, 1)
    names = [pred['input'] for pred in user_predictions]
    detections = [pred['detection'] for pred in user_predictions]
    
    bars = plt.bar(range(len(names)), detections, alpha=0.7)
    plt.axhline(y=baseline, color='red', linestyle='--', label=f'Current: {baseline:.3f}')
    plt.xlabel('Your Scenarios')
    plt.ylabel('Detection (µmol)')
    plt.title('Detection Comparison')
    plt.xticks(range(len(names)), [n[:10] for n in names], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency comparison
    plt.subplot(2, 2, 2)
    efficiencies = [pred['efficiency'] for pred in user_predictions]
    plt.bar(range(len(names)), efficiencies, alpha=0.7, color='green')
    plt.xlabel('Your Scenarios')
    plt.ylabel('Efficiency (%)')
    plt.title('Detection Efficiency')
    plt.xticks(range(len(names)), [n[:10] for n in names], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Ratio vs Detection
    plt.subplot(2, 2, 3)
    ratios = [pred['ratio'] for pred in user_predictions]
    plt.scatter(ratios, detections, s=100, alpha=0.7)
    for i, pred in enumerate(user_predictions):
        plt.annotate(pred['input'][:8], (pred['ratio'], pred['detection']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Ratio (V_solution/V_headspace)')
    plt.ylabel('Detection (µmol)')
    plt.title('Ratio vs Detection')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Improvement comparison
    plt.subplot(2, 2, 4)
    improvements = [pred['improvement'] for pred in user_predictions]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    plt.bar(range(len(names)), improvements, alpha=0.7, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Your Scenarios')
    plt.ylabel('Improvement (%)')
    plt.title('Improvement vs Current')
    plt.xticks(range(len(names)), [n[:10] for n in names], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Visual comparison generated!")

def find_optimal_ratio(fitted_params, baseline):
    """
    Automatically find optimal ratio
    """
    print(f"\n🔍 FINDING OPTIMAL RATIO...")
    
    # Test range of ratios
    ratios = np.linspace(0.3, 5.0, 50)
    total_vol = 0.008  # 8ml total
    
    best_detection = 0
    best_ratio = 0
    best_volumes = (0, 0)
    
    results = []
    
    for ratio in ratios:
        V_head = total_vol / (1 + ratio)
        V_sol = total_vol - V_head
        
        pred_n_gas, _ = predict_new_ratio(fitted_params, V_sol, V_head, np.array([100000]))
        detection = pred_n_gas[0]
        
        results.append({'ratio': ratio, 'detection': detection, 'V_sol': V_sol, 'V_head': V_head})
        
        if detection > best_detection:
            best_detection = detection
            best_ratio = ratio
            best_volumes = (V_sol, V_head)
    
    improvement = ((best_detection - baseline) / baseline) * 100
    
    print(f"🎯 OPTIMAL CONFIGURATION FOUND:")
    print(f"   Best ratio: {best_ratio:.2f}")
    print(f"   Volumes: {best_volumes[0]*1000:.1f}ml solution : {best_volumes[1]*1000:.1f}ml headspace")
    print(f"   Predicted detection: {best_detection:.6f} µmol")
    print(f"   Improvement: {improvement:+.1f}%")
    
    # Show top 5 alternatives
    sorted_results = sorted(results, key=lambda x: x['detection'], reverse=True)[:5]
    
    print(f"\n🏆 TOP 5 CONFIGURATIONS:")
    print(f"{'Rank':<6} {'Ratio':<8} {'Volumes (ml)':<15} {'Detection (µmol)':<15}")
    print("-" * 50)
    
    for i, result in enumerate(sorted_results, 1):
        vol_str = f"{result['V_sol']*1000:.1f}:{result['V_head']*1000:.1f}"
        print(f"{i:<6} {result['ratio']:<8.2f} {vol_str:<15} {result['detection']:<15.6f}")

# Update main function to include interactive mode
def main_henry_analysis(file_path, interactive=True):
    """
    Complete Henry's Law analysis pipeline with optional interactive mode
    """
    print("PHYSICS-BASED OXYGEN PARTITIONING ANALYSIS")
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
    
    print(f"✅ Model fitted successfully! (R² = {r2:.3f})")
    
    if interactive:
        # Launch interactive mode
        interactive_ratio_explorer(popt)
    else:
        # Run standard analysis
        from your_previous_functions import analyze_ratio_effects, plot_henry_model_results
        ratio_results = analyze_ratio_effects(popt)
        plot_henry_model_results(time_data, n_gas_measured, n_gas_predicted, popt, r2, rmse)

# Usage with interactive mode
if __name__ == "__main__":
    file_path = "M2_2.18mM_pressure_o2_release.csv"
    
    # Choose mode
    mode = input("Choose mode - (i)nteractive or (s)tandard analysis: ").strip().lower()
    
    if mode.startswith('i'):
        main_henry_analysis(file_path, interactive=True)
    else:
        main_henry_analysis(file_path, interactive=False)
