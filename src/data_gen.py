import numpy as np
import pandas as pd
from src.config import M_MODE, LAMBDA_REF

def generate_mrr_dataset(n_samples=1500, seed=42):
    np.random.seed(seed) # [cite: 8]
    
    # Design Parameters [cite: 10, 12, 15]
    R = np.random.uniform(5e-6, 50e-6, n_samples)
    w = np.random.uniform(300e-9, 800e-9, n_samples)
    g = np.random.uniform(50e-9, 500e-9, n_samples)
    
    # Physics Engine [cite: 18, 20, 23]
    L = 2 * np.pi * R
    n_eff = 2.1 + 0.5 * (w / 800e-9)
    lambda_res = (n_eff * L) / M_MODE
    
    # Q-Factor Calculation [cite: 25, 38]
    coupling = np.exp(-g / 350e-9)
    loss_w = 0.05 * (500e-9 / w)
    loss_R = 0.03 * (10e-6 / R)
    total_loss = loss_w + loss_R
    
    Q_base = (2 * np.pi * n_eff * L) / (LAMBDA_REF * (loss_w + loss_R + coupling))
    
    # Add Noise [cite: 40, 43]
    lambda_res += np.random.normal(0, lambda_res * 0.0005, n_samples)
    Q = Q_base + np.random.normal(0, Q_base * 0.02, n_samples)
    
    return pd.DataFrame({"R": R, "w": w, "g": g, "lambda_res": lambda_res, "Q": Q}) # [cite: 44-50]
