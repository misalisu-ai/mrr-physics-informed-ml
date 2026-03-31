import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import r2_score

def plot_exploratory_data(df):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # R vs Lambda_res 
    ax1.scatter(df["R"], df["lambda_res"], alpha=0.5, s=10)
    ax1.set_xlabel("Radius (R)")
    ax1.set_ylabel("Resonance Wavelength ($\lambda_{res}$)")
    ax1.set_title("R vs $\lambda_{res}$")
    ax1.grid(True, alpha=0.3)
    
    # Gap vs Q 
    ax2.scatter(df["g"], df["Q"], alpha=0.5, s=10, color='orange')
    ax2.set_xlabel("Gap (g)")
    ax2.set_ylabel("Q Factor")
    ax2.set_title("Gap vs Q")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_model_performance(y_actual, y_pred):
    
    r2 = r2_score(y_actual, y_pred)
    
    plt.figure(figsize=(7, 7))
    plt.scatter(y_actual, y_pred, alpha=0.4, color='teal', label='Predictions')
    
    # Perfect Fit Line
    lims = [np.min([y_actual.min(), y_pred.min()]), np.max([y_actual.max(), y_pred.max()])]
    plt.plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
    
    plt.xlabel("Actual Physical Q")
    plt.ylabel("Predicted Physical Q")
    plt.title(f"Model Success: $R^2 = {r2:.4f}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def plot_feature_importance(feat_imp):
    
    plt.figure(figsize=(10, 6))
    plt.bar(feat_imp["feature"], feat_imp["importance"], color='dodgerblue')
    plt.xticks(rotation=45)
    plt.title("Feature Importance (Physics-Informed)")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    return plt.gcf()

def plot_sensitivity_analysis(r_range, predictions):
    
    plt.figure(figsize=(8, 5))
 
    plt.plot(r_range * 1e6, predictions, color='dodgerblue', lw=2) 
    plt.xlabel("Radius ($\mu\text{m}$)")
    plt.ylabel("Predicted Q-Factor")
    plt.title("Physical Sensitivity Analysis: Radius vs Q")
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def plot_shap_explanation(shap_values, feature_names, index=0):
    
    plt.figure(figsize=(10, 6))
  
    shap.plots.waterfall(shap_values[index], max_display=10)
    return plt.gcf()
