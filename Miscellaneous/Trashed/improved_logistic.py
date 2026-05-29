import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

def load_pressure_data(file_path):
    df = pd.read_csv(file_path)
    clean_df = df.dropna()
    
    if 'DWT denoised pressure (kPa)' in clean_df.columns:
        clean_df['Pressure (atm)'] = clean_df['DWT denoised pressure (kPa)'] / 101.325
    
    return clean_df

#  (Don't change this!)
def improved_logistic_henry_model(time, P_total_max, k, t_lag, H, P_baseline,
                                 V_solution=0.006, V_headspace=0.002, R=0.08206, T=297.15):

    # Logistic growth for total pressure
    P_total = P_baseline + (P_total_max - P_baseline) / (1 + np.exp(-k * (time - t_lag)))   #modified
    
    # Henry's Law partitioning
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    
    # Detectable pressure
    P_gas = P_baseline + (P_total - P_baseline) * partition_fraction
    
    return P_gas

# ALTERNATIVE: Slightly modified for early points 
def early_improved_logistic_henry(time, P_total_max, k, t_lag, H, P_baseline, early_factor=1.0,
                                 V_solution=0.006, V_headspace=0.002, R=0.08206, T=297.15):
  
  
    # Logistic growth with optional early adjustment
    logistic_term = 1 / (1 + np.exp(-k * (time - t_lag)))
    
    # Optional: slower start for very early times
    if early_factor != 1.0:
        early_mask = time < t_lag
        early_adjustment = np.where(early_mask, 
                                  logistic_term ** early_factor,  # Slower early growth
                                  logistic_term)                 # Normal later growth
    else:
        early_adjustment = logistic_term
    
    P_total = P_baseline + (P_total_max - P_baseline) * early_adjustment
    
    # Henry's Law partitioning (same as before)
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    
    P_gas = P_baseline + (P_total - P_baseline) * partition_fraction
    
    return P_gas

def analyze_data_fit_quality(pressure_measured, time_data, prediction, model_name):
    """
    Detailed analysis of where the model fits well/poorly
    """
    residuals = pressure_measured - prediction
    
    # Calculate fit quality in different time regions
    total_time = time_data[-1] - time_data[0]
    
    # Define regions
    early_mask = time_data < (time_data[0] + total_time * 0.3)    # First 30%
    middle_mask = (time_data >= (time_data[0] + total_time * 0.3)) & \
                  (time_data < (time_data[0] + total_time * 0.7))   # Middle 40%
    late_mask = time_data >= (time_data[0] + total_time * 0.7)     # Last 30%
    
    # Calculate metrics for each region
    regions = {
        'Early': early_mask,
        'Middle': middle_mask, 
        'Late': late_mask
    }
    
    print(f"\n{model_name} - Regional Fit Analysis:")
    print("="*50)
    
    overall_r2 = r2_score(pressure_measured, prediction)
    overall_rmse = np.sqrt(mean_squared_error(pressure_measured, prediction))  # atm units
    
    print(f"Overall: R² = {overall_r2:.4f}, RMSE = {overall_rmse:.6f} atm")
    
    for region_name, mask in regions.items():
        if np.any(mask):
            region_r2 = r2_score(pressure_measured[mask], prediction[mask])
            region_rmse = np.sqrt(mean_squared_error(pressure_measured[mask], prediction[mask]))
            region_points = np.sum(mask)
            
            print(f"{region_name:>8}: R² = {region_r2:.4f}, RMSE = {region_rmse:.6f} atm ({region_points} points)")
    
    return {
        'overall_r2': overall_r2,
        'overall_rmse': overall_rmse,
        'residuals': residuals,
        'regions': regions
    }

def fit_optimal_models(df):
    """
    Fit your excellent model + small variations to see what works best
    """
    time_data = df['Time (s)'].values
    pressure_measured = df['Pressure (atm)'].values
    
    # Data analysis (same as before)
    baseline_points = int(len(pressure_measured) * 0.1)
    P_baseline = np.mean(pressure_measured[:baseline_points])
    plateau_points = int(len(pressure_measured) * 0.2)
    P_plateau = np.mean(pressure_measured[-plateau_points:])
    
    smoothed_data = np.convolve(pressure_measured - P_baseline, np.ones(5)/5, mode='same')
    derivatives = np.gradient(smoothed_data, time_data)
    inflection_idx = np.argmax(derivatives)
    t_inflection = time_data[inflection_idx]
    
    max_slope = np.max(derivatives)
    k_estimate = 4 * max_slope / (P_plateau - P_baseline)

    
    results = {}
    
    # MODEL 1: Your excellent original model
    try:
        P_total_guess = P_plateau * 2
        H_guess = 1.3e-3
        
        popt1, pcov1 = curve_fit(
            improved_logistic_henry_model, time_data, pressure_measured,
            p0=[P_total_guess, k_estimate, t_inflection, H_guess, P_baseline],
            bounds=(
                [P_plateau, k_estimate/10, t_inflection*0.5, 5e-4, P_baseline*0.5],
                [P_plateau*5, k_estimate*10, t_inflection*2, 3e-3, P_baseline*2]
            ),
            maxfev=15000
        )
        
        pred1 = improved_logistic_henry_model(time_data, *popt1)
        
        results['excellent_original'] = {
            'params': popt1,
            'prediction': pred1,
            'model_name': 'improved Model',
            'param_names': ['P_total_max', 'k', 't_lag', 'H', 'P_baseline']
        }
        
        # Detailed analysis
        analysis1 = analyze_data_fit_quality(pressure_measured, time_data, pred1, 
                                           "Your Excellent Model")
        results['excellent_original']['analysis'] = analysis1
        
    except Exception as e:
        print(f"Excellent model fitting failed: {e}")
        results['excellent_original'] = None
    
    # MODEL 2: Minor variation (only if original has issues in early region)
    try:
        if results['excellent_original'] is not None:
            early_r2 = None
            for region, mask in results['excellent_original']['analysis']['regions'].items():
                if region == 'Early' and np.any(mask):
                    early_r2 = r2_score(pressure_measured[mask], pred1[mask])
                    break
            
            # Only try variation if early fit is poor
            if early_r2 is not None and early_r2 < 0.95:
                print(f"\nTrying minor variation for early fit improvement...")
                
                popt2, pcov2 = curve_fit(
                    early_improved_logistic_henry, time_data, pressure_measured,
                    p0=[P_total_guess, k_estimate, t_inflection, H_guess, P_baseline, 1.2],
                    bounds=(
                        [P_plateau, k_estimate/10, t_inflection*0.5, 5e-4, P_baseline*0.5, 0.5],
                        [P_plateau*5, k_estimate*10, t_inflection*2, 3e-3, P_baseline*2, 2.0]
                    ),
                    maxfev=15000
                )
                
                pred2 = early_improved_logistic_henry(time_data, *popt2)
                
                results['minor_variation'] = {
                    'params': popt2,
                    'prediction': pred2,
                    'model_name': 'Minor Early Adjustment model',
                    'param_names': ['P_total_max', 'k', 't_lag', 'H', 'P_baseline', 'early_factor']
                }
                
                analysis2 = analyze_data_fit_quality(pressure_measured, time_data, pred2,
                                                   "Minor Early Adjustment")
                results['minor_variation']['analysis'] = analysis2
            else:
                print(f"\nEarly fit is excellent (R² = {early_r2:.3f}) - no variation needed!")
                
    except Exception as e:
        print(f"Minor variation failed: {e}")
        results['minor_variation'] = None
    
    return results, time_data, pressure_measured

def plot_optimal_comparison(results, time_data, pressure_measured):
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#2ECC71', '#E74C3C']  # Green for excellent, red for alternative
    
    # Plot 1: Full comparison
    ax1.scatter(time_data/3600, pressure_measured, alpha=0.8, s=30, 
               color='black', label='Experimental Data', zorder=5, edgecolors='black')
    
    for i, (model_name, result) in enumerate(results.items()):
        if result is not None:
            r2 = result['analysis']['overall_r2']
            rmse = result['analysis']['overall_rmse']
            
            ax1.plot(time_data/3600, result['prediction'], 
                    color=colors[i], linewidth=3, alpha=0.9,
                    label=f"{result['model_name']} (R²={r2:.3f}, RMSE={rmse:.6f} atm)")
    
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pressure (atm)', fontsize=12, fontweight='bold')
    ax1.set_title('Improved Simple Logistics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Early phase zoom
    early_hours = 0.03  # First 0.03 hours
    early_mask = time_data <= early_hours * 3600
    
    ax2.scatter(time_data[early_mask]/3600, pressure_measured[early_mask], 
               alpha=0.8, s=40, color='black', label='Experimental Data', zorder=5)
    
    for i, (model_name, result) in enumerate(results.items()):
        if result is not None:
            ax2.plot(time_data[early_mask]/3600, result['prediction'][early_mask], 
                    color=colors[i], linewidth=3, alpha=0.9,
                    label=f"{result['model_name']}")
    
    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pressure (atm)', fontsize=12, fontweight='bold')
    ax2.set_title('Early Phase Detail', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    

    plt.tight_layout()
    plt.show()

def main_optimal_analysis():
    
    file_path = "M2_2.18mM_pressure_o2_release.csv"
    df = load_pressure_data(file_path)
    
    results, time_data, pressure_measured = fit_optimal_models(df)
    
    plot_optimal_comparison(results, time_data, pressure_measured)
    
    return results

if __name__ == "__main__":
    results = main_optimal_analysis()

