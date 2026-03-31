from src.data_gen import generate_mrr_dataset
from src.features import create_features
from src.model import get_model_pipeline, run_inverse_design
import numpy as np

# 1. Generate Data
df = generate_mrr_dataset()
df = create_features(df)

# 2. Prepare Training
features = ["R", "w", "g", "L", "FSR", "g_norm", "coupling_ratio", "A_eff", "AR"] 
X = df[features]
y_log = np.log10(df["Q"]) 

# 3. Train
pipeline = get_model_pipeline()
pipeline.fit(X, y_log) 

# 4. Inverse Design Example
optimal_geom = run_inverse_design(pipeline, 50000, features)
print(f"Optimal Design: R={optimal_geom*1e6:.2f}um, w={optimal_geom[1]*1e9:.2f}nm, g={optimal_geom[2]*1e9:.2f}nm")
