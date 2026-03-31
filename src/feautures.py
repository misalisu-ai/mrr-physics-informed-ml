import numpy as np
from src.config import H_CHIP, C_LIGHT, N_GROUP

def create_features(df):
    df = df.copy() 
    
    # Optical & Coupling Features 
    df["L"] = 2 * np.pi * df["R"]
    df["FSR"] = C_LIGHT / (N_GROUP * df["L"])
    df["g_norm"] = df["g"] / df["w"]
    df["coupling_ratio"] = df["w"] / df["g"]
    df["A_eff"] = df["w"] * H_CHIP
    df["AR"] = df["w"] / H_CHIP
    
    return df
