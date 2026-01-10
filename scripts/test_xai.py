import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from models.explainer import IDSExplainer
from models.wrapper import IDSModel
from utils.logger import setup_logging

def test_xai():
    setup_logging()
    print("Testing XAI Module...")
    
    # 1. Load Model (Mock or Real)
    # We'll try to load the advanced model if it exists, else mock it
    model_path = "models/artifacts/model_advanced.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Loaded Real Advanced Model.")
    else:
        print("Warning: Real model not found, skipping specific integration test.")
        return

    # 2. Mock Data (Processed)
    # Must match training feature names exactly
    # We attempt to load the preprocessor to get feature names if possible or guess
    cols = ['num__length_log', 'cat__src_ip_10.0.0.2', 'cat__src_ip_172.16.0.1', 
            'cat__src_ip_192.168.1.5', 'cat__src_ip_192.168.1.6', 
            'cat__dst_ip_10.0.0.5', 'cat__dst_ip_8.8.8.8', 
            'cat__protocol_ICMP', 'cat__protocol_TCP', 'cat__protocol_UDP', 
            'cat__flags_ACK', 'cat__flags_FIN', 'cat__flags_PSH', 
            'cat__flags_RST', 'cat__flags_SYN']
    
    # Create single row
    # Just correct shape
    row_data = np.random.rand(1, 15) 
    row_df = pd.DataFrame(row_data, columns=cols) # Using dummy columns matching shape roughly
    
    # Create background sample (necessary for fallback)
    background_df = pd.DataFrame(np.random.rand(20, 15), columns=cols)
    
    # 3. Initialize Explainer
    explainer = IDSExplainer(model, X_sample=background_df)
    
    # 4. Explain Local
    print("\n--- Local Explanation ---")
    # We pass the row_df. Ideally it should match model input.
    # To avoid mismatch error, we can try to extract names from model or use what we have
    # If using XGBoost sklearn API:
    try:
        booster = model.get_booster()
        actual_features = booster.feature_names
        # Re-create df with correct names
        if actual_features:
            row_df = pd.DataFrame(row_data, columns=actual_features)
            background_df = pd.DataFrame(np.random.rand(20, len(actual_features)), columns=actual_features)
    except:
        pass
        
    # Re-init with correct columns if changed
    explainer = IDSExplainer(model, X_sample=background_df)
    explanation = explainer.explain_local(row_df)
    import json
    print(json.dumps(explanation, indent=2))
    
    # 5. Global Plots (Mock Dataset)
    print("\n--- Global Plots ---")
    mock_batch = pd.DataFrame(np.random.rand(50, len(row_df.columns)), columns=row_df.columns)
    explainer.explain_global(mock_batch)
    
if __name__ == "__main__":
    test_xai()
