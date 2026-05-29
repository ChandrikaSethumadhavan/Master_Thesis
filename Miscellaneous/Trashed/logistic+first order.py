#Simplified Logistic-Henry Model with Automatic Plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

C0 = 2.18e-3  # mol/L, for M2 (from protocol)

def load_pressure_data(file_path):
    
        df = pd.read_csv(file_path)
        clean_df = df.dropna()

        return clean_df
        
    

def logistic_henry_model(time, n_total_max, growth_rate, lag_time, H, V_solution = 0.006, V_headspace = 0.002, R=0.08206, T=310.15): #Vsol and headspace will change depemdnding on ip

   # to handle reaction kinetics
    n_total = n_total_max / (1 + np.exp(-growth_rate * (time - lag_time))) #logistic formula models total O2 released over time using logistic growth
    
    # to handle thermodynamics
    denominator = (V_headspace / (R * T)) + (H * V_solution)  #henry's law 
    partition_fraction = (V_headspace / (R * T)) / denominator
    
    # deetectable O₂ in headspace
    n_gas = n_total * partition_fraction  #logistic x henry model
    
    #print("n_gas values", n_gas[0])  # Debugging output
    return n_gas

def first_order_henry_model(time, n_total_max, rate_constant, H, V_solution=0.006, V_headspace=0.002, R=0.08206, T=310.15):
    n_total = n_total_max * (1 - np.exp(-rate_constant * time))  # first-order reaction kinetics
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    n_gas = n_total * partition_fraction
    return n_gas



def fit_models(df, V_solution, V_headspace):
    
    #Uses nonlinear least squares (Levenberg-Marquardt) to fit combined model to real pressure/O2 data.
    
    time_data = df['Time (s)'].values
    n_gas_measured = df['O2 Released (µmol)'].values  # This is what pressure sensor detects               PHYSICAL
    n_total_theoretical = C0 * V_solution * 1e6  # µmol # Maximum possible from stoichiometry      THEORETICAL
    print(f"Total theoretical O₂: {n_total_theoretical:.2f} µmol")  #test
    
    # Initial parameter estimation with debugging
    max_measured = n_gas_measured.max()  # Maximum measured O₂ released physically for M2 was 4.82 ish 
    

    # to override the problem of oxygen dissolvability for now, we will use a more conservative estimate for n_total_max

    # 
    estimated_n_total_max = min(max_measured * 3, n_total_theoretical * 0.95)

   # print(f"Estimated n_total_max: {estimated_n_total_max:.2f} µmol")  # Debugging output
    
    
# K growth rate estimation
    slopes = np.gradient(n_gas_measured, time_data)
    max_slope = np.max(slopes)
    estimated_growth_rate = (4 * max_slope) / estimated_n_total_max
    estimated_growth_rate *= 4

    
    # lag time estimation
    # Find point closet to 50% of max
    target_value = max_measured * 0.5
    lag_idx = np.argmin(np.abs(n_gas_measured - target_value))
    estimated_lag_time = time_data[lag_idx]
  
    
    # Henry's constant from your experimental data
    initial_H = 0.0221
  

    initial_params = [estimated_n_total_max, estimated_growth_rate, estimated_lag_time, initial_H]
    
    #define the flexible bounds
    lower_bounds = [
        max_measured,                    # n_total_max must be at least max measured
        1e-8,                           # very small growth rate
        0,                              # lag can start immediately
        1e-6                            # very small Henry's constant
    ]
    
    upper_bounds = [
        n_total_theoretical * 5,        # generous upper limit
        1e-1,                          # reasonable upper growth rate
        time_data.max(),               # lag can be anywhere in time range
        1.0                            # generous Henry's constant upper limit
    ]
    
    #Ensure initial params are within bounds  (guess into the valid zone by ±10%  !!!!!!!)
    for i in range(len(initial_params)):
        if initial_params[i] < lower_bounds[i]:
            initial_params[i] = lower_bounds[i] * 1.1
        if initial_params[i] > upper_bounds[i]:
            initial_params[i] = upper_bounds[i] * 0.9
    
    
    # Try fitting with multiple strategies . Nonlinear fitting is sensitive to bad initial guesses. Multiple strategies improve robustness
    fitting_strategies = [
        # Strategy 1: Original bounds
        (initial_params, lower_bounds, upper_bounds),
        
        # Strategy 2
        (initial_params, None, None), # Same starting guess as Strategy 1, But removes all bounds
        
        # Strategy 3: tarts from a very simple, generic guess but with bounds
        ([max_measured * 2, 1e-5, time_data.max()/2, 0.02], lower_bounds, upper_bounds)

    ]

    
    
    for strategy_num, (params, lower, upper) in enumerate(fitting_strategies, 1):
       
            print(f" Trying fitting strategy {strategy_num}...")

            model = lambda t, n_tot, k, t_lag, H: logistic_henry_model(t, n_tot, k, t_lag, H, V_solution=V_solution, V_headspace=V_headspace)
            
            if lower is not None and upper is not None:
                popt, pcov = curve_fit(         #  popt the best-fit parameters
                    model, time_data, n_gas_measured,
                    p0=params, bounds=(lower, upper), maxfev=15000
                )
        
            
            # Calculate fit quality
            n_gas_predicted = logistic_henry_model(time_data, *popt)  #This simulates the oxygen release using the best-fit parameters.
            #return popt, time_data, n_gas_measured, n_gas_predicted
            
            
    def first_order_model(t, n_tot, k, H):
        return first_order_henry_model(t, n_tot, k, H, V_solution=V_solution, V_headspace=V_headspace)

    p0_fo = [estimated_n_total_max, 1e-5, initial_H]
    bounds_fo = ([max_measured, 1e-8, 1e-6], [n_total_theoretical * 5, 1e-1, 1.0])
    popt_fo, _ = curve_fit(first_order_model, time_data, n_gas_measured, p0=p0_fo, bounds=bounds_fo)

    n_gas_pred_fo = first_order_model(time_data, *popt_fo)

    from sklearn.metrics import r2_score, mean_squared_error
    r2_log = r2_score(n_gas_measured, n_gas_predicted)
    rmse_log = np.sqrt(mean_squared_error(n_gas_measured, n_gas_predicted))

    r2_fo = r2_score(n_gas_measured, n_gas_pred_fo)
    rmse_fo = np.sqrt(mean_squared_error(n_gas_measured, n_gas_pred_fo))

    print("\n--- Model Fit Comparison ---")
    print(f"Logistic-Henry:     R² = {r2_log:.4f}, RMSE = {rmse_log:.4f}")
    print(f"First-Order-Henry:  R² = {r2_fo:.4f}, RMSE = {rmse_fo:.4f}")

    return {
        'logistic': (popt, time_data, n_gas_measured, n_gas_predicted, r2_log, rmse_log),
        'first_order': (popt_fo, time_data, n_gas_measured, n_gas_pred_fo, r2_fo, rmse_fo)
    }
        
            
        


def get_user_volume_setups():
    
    
    print("\n" + "="*60)
    print("📝 ENTER YOUR VOLUME RATIOS TO TEST")
    print("="*60)
    
    
    volume_setups = []
    setup_counter = 1
    
    while True:
        user_input = input(f"\nSetup {setup_counter}: ").strip()
        
        # Exit condition
        if user_input.lower() in ['done', 'exit', 'quit', '']:
            break
        
        
        # Parse different input formats
        if ':' in user_input:
            # Format: "8:4" or "8ml:4ml"
            parts = user_input.replace('ml', '').replace(' ', '').split(':')
            if len(parts) == 2:
                V_sol = float(parts[0])
                V_head = float(parts[1])
                name = f"Setup {setup_counter}: {V_sol}ml:{V_head}ml"
                volume_setups.append((V_sol, V_head, name))
                print(f"✅ Added: {name} (ratio = {V_sol/V_head:.2f})")
                setup_counter += 1
            else:
                print("Invalid format. Use 'solution:headspace' (e.g., '8:4')")
    
    
    print(f"\n✅ Total setups to test: {len(volume_setups)}")
    return volume_setups


    
  

def predict_and_plot_enhanced(fitted_params, V_solution, V_headspace, setup_name="", baseline_params=None):
    
    
    # Scale parameters for new volume
    old_V_solution = 0.006
    fitted_n_total_max, fitted_growth_rate, fitted_lag_time, fitted_H = fitted_params
    scaled_n_total_max = fitted_n_total_max * (V_solution / old_V_solution)
    
    # Generate time points for prediction
    time_points = np.linspace(0, 200000, 1000)  # 0 to ~55 hours
    
    # Predict with new volumes
    predicted_n_gas = logistic_henry_model(
        time_points, scaled_n_total_max, fitted_growth_rate, fitted_lag_time, fitted_H,
        V_solution=V_solution, V_headspace=V_headspace
    )
    
    # Calculate metrics
    ratio = V_solution / V_headspace
    final_detection = predicted_n_gas[-1]  # At equilibrium
    
    # Calculate detection efficiency
    R, T = 0.08206, 310.15
    denominator = (V_headspace / (R * T)) + (fitted_H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    efficiency = partition_fraction * 100
    
    # Compare to baseline (6ml:2ml)
    baseline_detection = logistic_henry_model(
        np.array([200000]), fitted_n_total_max, fitted_growth_rate, fitted_lag_time, fitted_H
    )[0]
    improvement = ((final_detection - baseline_detection) / baseline_detection) * 100
    
    
    
    # Quality assessment
    if improvement > 25:
        quality = " EXCELLENT"
        color_main = 'green'
    elif improvement > 10:
        quality = " GOOD"
        color_main = 'blue'
    elif improvement > -5:
        quality = " SIMILAR"
        color_main = 'orange'
    else:
        quality = " POOR"
        color_main = 'red'
    
    
    
    # Generate simplified 3-panel plot
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{setup_name}\nRatio: {ratio:.2f}, Efficiency: {efficiency:.1f}%, Improvement: {improvement:+.1f}%', 
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Improvement comparison
    plt.subplot(1, 3, 1)
    scenarios = ['Baseline\n(6ml:2ml)', f'Setup\n({V_solution*1000:.1f}:{V_headspace*1000:.1f}ml)']
    detections = [baseline_detection, final_detection]
    colors = ['gray', color_main]
    
    bars = plt.bar(scenarios, detections, color=colors, alpha=0.7)
    plt.ylabel('Final Detection (µmol)')
    plt.title(f'Detection Comparison\n{improvement:+.1f}% change')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, detections):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(detections)*0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 2: Efficiency vs ratio optimization
    plt.subplot(1, 3, 2)
    test_ratios = np.linspace(0.5, 5.0, 50)
    test_efficiencies = []
    
    for test_ratio in test_ratios:
        total_vol = V_solution + V_headspace
        test_V_head = total_vol / (1 + test_ratio)
        test_V_sol = total_vol - test_V_head
        
        test_denom = (test_V_head / (R * T)) + (fitted_H * test_V_sol)
        test_eff = ((test_V_head / (R * T)) / test_denom) * 100
        test_efficiencies.append(test_eff)
    
    plt.plot(test_ratios, test_efficiencies, 'purple', linewidth=2, alpha=0.7)
    plt.axvline(x=ratio, color=color_main, linestyle='--', linewidth=2, 
                label=f'Your ratio: {ratio:.2f}')
    plt.axhline(y=efficiency, color=color_main, linestyle='--', linewidth=2, alpha=0.7)
    plt.scatter([ratio], [efficiency], color=color_main, s=150, zorder=5, edgecolor='black')
    
    # Mark baseline
    baseline_ratio = 3.0  # 6ml:2ml
    baseline_eff_idx = np.argmin(np.abs(test_ratios - baseline_ratio))
    baseline_eff = test_efficiencies[baseline_eff_idx]
    plt.scatter([baseline_ratio], [baseline_eff], color='gray', s=100, zorder=5, 
                edgecolor='black', label='Baseline')
    
    plt.xlabel('V_solution/V_headspace Ratio')
    plt.ylabel('Detection Efficiency (%)')
    plt.title('Efficiency vs Volume Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 3: Summary statistics
    plt.subplot(1, 3, 3)
    plt.axis('off')
    
    # Calculate additional metrics
    max_rate = np.max(np.gradient(predicted_n_gas, time_points) * 3600)
    total_volume = V_solution + V_headspace
    
    # Time to reach milestones (for summary only)
    milestones = [0.9]  # Just 90% for summary
    target = final_detection * milestones[0]
    idx = np.argmin(np.abs(predicted_n_gas - target))
    
    stats_text = f"""
SUMMARY STATISTICS

Volume Setup:
• Solution: {V_solution*1000:.1f} ml
• Headspace: {V_headspace*1000:.1f} ml  
• Total: {total_volume*1000:.1f} ml
• Ratio: {ratio:.2f}

Performance:
• Final Detection: {final_detection:.4f} µmol
• Efficiency: {efficiency:.1f}%
• Improvement: {improvement:+.1f}%
• Assessment: {quality}

"""
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
 

def main_analysis():
    
    
    
    file_path = "M2_2.18mM_pressure_o2_release.csv"  
    df = load_pressure_data(file_path)
    
    volume_setups = get_user_volume_setups()

    # Use the first setup to fit the model
    V_sol_ml, V_head_ml, _ = volume_setups[0]   # Just use the first one for fitting
    V_solution = V_sol_ml / 1000
    V_headspace = V_head_ml / 1000

    fit_results_dict = fit_models(df, V_solution, V_headspace)
    log_fit = fit_results_dict['logistic']
    fo_fit = fit_results_dict['first_order']

    # fitted_params, time_data, n_gas_measured, n_gas_predicted = fit_results_dict
    popt_fo, time_data_fo, n_gas_measured_fo, n_gas_pred_fo, r2_fo, rmse_fo = fit_results_dict['first_order']

    plt.figure(figsize=(10, 5))
    plt.plot(log_fit[1]/3600, log_fit[2], 'k.', label='Measured Data')
    plt.plot(log_fit[1]/3600, log_fit[3], 'g-', label='Logistic Fit')
    plt.plot(fo_fit[1]/3600, fo_fit[3], 'r--', label='First-Order Fit')
    plt.xlabel('Time (hours)')
    plt.ylabel('O₂ Released (µmol)')
    plt.title('Model Comparison: Logistic vs First-Order')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    
    
    
    
    
    # Run predictions with user setups
    
    results = []
    
    for i, (V_sol, V_head, name) in enumerate(volume_setups):
        print(f"\n{'='*60}")
        print(f"PREDICTION {i+1}/{len(volume_setups)}: {name}")
        print(f"{'='*60}")
        
        result = predict_and_plot_enhanced(fitted_params, V_sol/1000, V_head/1000, name)
        if result:
            results.append({**result, 'name': name, 'V_sol': V_sol, 'V_head': V_head})
    
  
        
        
if __name__ == "__main__":
    main_analysis()