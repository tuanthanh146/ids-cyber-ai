import pandas as pd
import numpy as np
import joblib
import yaml
import os
import logging
from sklearn.ensemble import IsolationForest
from datetime import datetime

logger = logging.getLogger(__name__)

class AnomalyTrainer:
    def __init__(self, config_path="configs/retrain_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.ad_config = self.config.get("anomaly_detection", {})
        self.paths = self.config["retraining"]["paths"]
        
    def load_recent_normal_traffic(self, days=14):
        import sys
        sys.path.append(os.getcwd())
        from utils.data_manager import DataManager
        dm = DataManager()
        # Load strictly normal traffic for IF training
        return dm.load_recent_normal(days=days)

    def determine_threshold(self, model, X_val, percentile=99):
        """
        Dynamic thresholding based on score percentiles.
        IsolationForest decision_function: average anomaly score of X of the base classifiers.
        The anomaly score of an input sample is computed as the mean anomaly score of the trees in the forest.
        The measure of normality of an observation given a tree is the depth of the leaf containing this observation, 
        which is equivalent to the number of splittings required to isolate this point.
        In case of several trees, it is the mean of the depth of the leaves.
        
        Scikit-learn: decision_function returns negative outlier score. 
        Lower = more anomalous.
        """
        scores = model.decision_function(X_val)
        
        # We want to flag the bottom (100-percentile)% as anomalies (if we assume contamination)
        # Or if we assume X_val is pure normal, we want threshold to cover 99% of it.
        # Threshold = Percentile(scores, 100-p) 
        # e.g. p=99 -> 1st percentile of scores. Anything lower is outlier.
        thresh = np.percentile(scores, 100 - percentile)
        return thresh

    def retrain_anomaly_model(self):
        logger.info("Starting Anomaly Model Retraining...")
        
        # 1. Load Data
        days = self.ad_config.get("train_window_days", 14)
        df = self.load_recent_normal_traffic(days=days)
        
        if df.empty:
            logger.error("No recent normal data found for anomaly training.")
            return None
            
        logger.info(f"Loaded {len(df)} samples for training.")
        
        # 2. Features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drops = ['label', 'attack_type', 'ts', 'src_port', 'dst_port'] # Drop metadata/labels
        feats = [c for c in numeric_cols if c not in drops and not c.startswith("_")]
        
        X_train = df[feats].fillna(0)
        
        # 3. Train Isolation Forest
        contam = self.ad_config.get("contamination", 0.01)
        model = IsolationForest(
            n_estimators=100,
            contamination=contam, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train)
        
        # 4. Dynamic Thresholding
        # Evaluate on a hold-out normal set? Or same set (if large enough).
        # Using same set for thresholding usually fine for unsupervised IF if we assume contamination.
        pct = self.ad_config.get("percentile_threshold", 99)
        threshold = self.determine_threshold(model, X_train, percentile=pct)
        logger.info(f"Dynamic Threshold (p={pct}): {threshold:.4f}")
        
        # 5. Save
        bundle = {
            "model": model,
            "threshold": threshold,
            "features": feats,
            "trained_at": datetime.now().isoformat()
        }
        
        out_path = os.path.join(self.paths["model_registry"], "anomaly_model.joblib")
        joblib.dump(bundle, out_path)
        logger.info(f"Anomaly Model Saved to {out_path}")
        
        return out_path, threshold

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = AnomalyTrainer()
    trainer.retrain_anomaly_model()
