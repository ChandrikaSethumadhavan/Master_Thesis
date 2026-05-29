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


# max theoretical o2 is p_total_mx

def improved_logistic_henry_model(time, P_total_max, k, t_lag, H, P_baseline,
                                 V_solution=0.006, V_headspace=0.002, R=0.08206, T=297.15):
    # Logistic growth for total pressure
    P_total = P_baseline + (P_total_max - P_baseline) / (1 + np.exp(-k * (time - t_lag)))
    
    # Henry's Law partitioning
    denominator = (V_headspace / (R * T)) + (H * V_solution)
    partition_fraction = (V_headspace / (R * T)) / denominator
    
    # Detectable pressure
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
    
    time_data = df['Time (s)'].values
    pressure_measured = df['Pressure (atm)'].values
    

    baseline_points = int(len(pressure_measured) * 0.1)  #creating a baseline by taking the mean value of first 10 % datapoints
    P_baseline = np.mean(pressure_measured[:baseline_points])  
    plateau_points = int(len(pressure_measured) * 0.2)
    P_plateau = np.mean(pressure_measured[-plateau_points:]) #final pressure is the mean of the last 20 points
    
    #smoothed_data = np.convolve(pressure_measured - P_baseline, np.ones(5)/5, mode='same')  # smoothing the curve further
    derivatives = np.gradient(pressure_measured, time_data)
    inflection_idx = np.argmax(derivatives) #wkt infletion point is the max slope point
    t_inflection = time_data[inflection_idx] #t5aking that time
    
    max_slope = np.max(derivatives)
    k_estimate = 4 * max_slope / (P_plateau - P_baseline)

    results = {}
    
    # MODEL 1
   
    P_total_guess = df["Pressure (atm)"].max()
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
        'param_names': ['P_total_max', 'k', 't_lag', 'H', 'P_baseline'],
        'k_value': popt1[1]  # Extract k value (second parameter)
    }
    
    # Detailed analysis
    analysis1 = analyze_data_fit_quality(pressure_measured, time_data, pred1, 
                                        "Your Excellent Model")
    results['excellent_original']['analysis'] = analysis1

    return results, time_data, pressure_measured



def plot_optimal_comparison(results, time_data, pressure_measured):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#2ECC71', '#E74C3C'] 
    ax1.scatter(time_data/3600, pressure_measured, alpha=0.8, s=30, 
               color='black', label='Experimental Data', zorder=5, edgecolors='black')
    for i, (model_name, result) in enumerate(results.items()):
        if result is not None:
            r2 = result['analysis']['overall_r2']
            rmse = result['analysis']['overall_rmse']
            k_val = result['k_value']
            
            ax1.plot(time_data/3600, result['prediction'], 
                    color=colors[i], linewidth=3, alpha=0.9,
                    label=f"{result['model_name']} (k={k_val:.2e} s⁻¹, R²={r2:.3f})")
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pressure (atm)', fontsize=12, fontweight='bold')
    ax1.set_title('Improved Simple Logistics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    







def main_optimal_analysis():
    
    file_path = "M2_2.18mM_pressure_o2_release.csv"
    df = load_pressure_data(file_path)
    
    results, time_data, pressure_measured = fit_optimal_models(df)
    
    plot_optimal_comparison(results, time_data, pressure_measured)
    
    for model_name, result in results.items():
        if result is not None:
            k_val = result['k_value']
            print(f"{result['model_name']}: k = {k_val:.4e} s⁻¹")
    
    return results

if __name__ == "__main__":
    results = main_optimal_analysis()
