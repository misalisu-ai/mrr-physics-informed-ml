import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
from src.features import create_features

def get_model_pipeline():
    # [cite: 153, 154]
    return Pipeline([
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42))
    ])

def run_inverse_design(model, target_Q, features_list):
    target_Q_log = np.log10(target_Q) # [cite: 332]
    
    def objective(x):
        # x = [R, w, g] [cite: 335]
        temp_df = pd.DataFrame([x], columns=['R', 'w', 'g'])
        temp_df = create_features(temp_df)
        pred_Q_log = model.predict(temp_df[features_list]) # [cite: 343]
        return (pred_Q_log - target_Q_log)**2 # [cite: 344]

    # [cite: 346, 347]
    res = minimize(objective, x0=[25e-6, 500e-9, 200e-9], 
                   bounds=[(5e-6, 50e-6), (300e-9, 800e-9), (50e-9, 500e-9)])
    return res.x
