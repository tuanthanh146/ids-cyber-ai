import argparse
import sys
import os
import yaml
import logging
import pandas as pd
import joblib
import shutil
import subprocess
import json
import time
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Add project root to path
sys.path.append(os.getcwd())

from utils.data_manager import DataManager
from utils.drift_detector import DriftDetector

# Setup Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/retrain_pipeline.log")
    ]
)
logger = logging.getLogger("RetrainPipeline")

class MLOpsPipeline:
    def __init__(self, config_path="configs/retrain_config.yaml"):
        self.config = self._load_config(config_path)
        self.dm = DataManager(config_path)
        self.drift_detector = DriftDetector(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths = self.config['retraining']['paths']
        
        # Ensure Dirs
        for p in [self.paths['model_registry'], self.paths['archive_dir'], "data/storage/temp"]:
            os.makedirs(p, exist_ok=True)

    def _load_config(self, path):
        with open(path, "r") as f: return yaml.safe_load(f)

    def check_drift(self):
        """Step 3: Drift Detection"""
        logger.info("--- Step 3: Drift Detection ---")
        ref_df = self.dm.load_benchmark()
        curr_df = self.dm.load_recent_normal(days=3)
        
        score, report = self.drift_detector.detect_drift(ref_df, curr_df)
        threshold = self.config['retraining']['drift_detection'].get('threshold_score', 0.5)
        
        logger.info(f"Drift Score: {score:.4f} (Threshold: {threshold})")
        
        # Save Drift Report
        self._save_json("drift_report.json", report)
        
        return score > threshold

    def prepare_data(self):
        """Step 4: Build Dataset & Zero-day Split"""
        logger.info("--- Step 4: Build Retrain Dataset ---")
        full_df = self.dm.build_retrain_dataset()
        
        if full_df.empty:
            raise ValueError("Empty dataset generated.")
            
        # Zero-day Simulation
        zero_day_df = pd.DataFrame()
        train_df = full_df.copy()
        
        holdout_col = 'attack_type' if 'attack_type' in full_df.columns else 'label'
        
        # Check if column exists and is not binary (numeric)
        is_numeric = pd.api.types.is_numeric_dtype(full_df[holdout_col])
        if is_numeric:
             # If label is numeric (0/1), we can't easily distinguish attack types for Zero-Day
             # Unless we have a separate attack_type column.
             if 'attack_type' in full_df.columns:
                 holdout_col = 'attack_type'
             else:
                 logger.warning("Cannot perform Zero-day holdout: No descriptive attack_type column found.")
                 holdout_col = None

        if holdout_col:
            counts = full_df[holdout_col].value_counts()
            # Normalize keys to string for comparison
            # Exclude Normal/Benign/0
            candidates = [
                k for k in counts.index 
                if str(k).lower() not in ['normal', 'benign', '0'] and counts[k] > 50
            ]
            
            # key requirement: Must have at least 2 attack types to hold one out, 
            # otherwise we remove all attacks and can't train binary classifier.
            if len(candidates) >= 2:
                holdout_type = candidates[-1] # Pick the last one
                
                mask = full_df[holdout_col] == holdout_type
                zero_day_df = full_df[mask].copy()
                train_df = full_df[~mask].copy()
                logger.info(f"Zero-day Simulation: Holding out '{holdout_type}' ({len(zero_day_df)} samples)")
            elif len(candidates) == 1:
                logger.info(f"Skipping Zero-day holdout: Only one attack type found ('{candidates[0]}'). Keeping it for training.")
        
        # Save temps
        train_path = f"data/storage/temp/train_{self.timestamp}.csv"
        zeroday_path = f"data/storage/temp/zeroday_{self.timestamp}.csv" if not zero_day_df.empty else None
        
        train_df.to_csv(train_path, index=False)
        if zeroday_path: zero_day_df.to_csv(zeroday_path, index=False)
        
        # Report
        report = {
            "train_size": len(train_df),
            "zero_day_holdout": holdout_type,
            "zero_day_size": len(zero_day_df)
        }
        self._save_json("dataset_report.json", report)
        
        return train_path, zeroday_path

    def train_model(self, train_path, test_path):
        """Step 5: Train"""
        logger.info("--- Step 5: Training Model ---")
        exp_dir = f"experiments/run_{self.timestamp}"
        
        cmd = [
            sys.executable, "scripts/train_smart.py",
            "--train_csv", train_path,
            "--test_csv", test_path, 
            "--outdir", exp_dir,
            "--tune", "0", # Faster for demo, set 1 for prod
            "--top_k", "20"
        ]
        
        subprocess.run(cmd, check=True)
        return os.path.join(exp_dir, "models", "ensemble_bundle.joblib")

    def evaluate(self, model_path, test_path, zeroday_path=None):
        """Step 6 & 7: Evaluation & Gate"""
        logger.info("--- Step 6: Evaluation ---")
        
        # 1. Benchmark Test (Golden Set)
        metrics_bench = self._eval_single(model_path, test_path, "Benchmark")
        
        # 2. Zero-day Test
        metrics_zd = {}
        if zeroday_path:
            # For zero-day, we expect the model to classify it as 'Attack' (Binary) 
            # or ideally 'Unknown' if we had open-set recognition.
            # Here we check Recall: does it detect it as MALICIOUS?
            metrics_zd = self._eval_single(model_path, zeroday_path, "ZeroDay")
        
        # 3. Canary Evaluation (Safety Gate)
        metrics_canary = {}
        canary_passed = True
        canary_path = self.config['retraining']['evaluation'].get('canary_path')
        if self.config['retraining']['data_mixing']['safety'].get('enable_canary_eval', False) and os.path.exists(canary_path):
             metrics_canary = self._eval_single(model_path, canary_path, "Canary")
             
             # Strict Gate: F1 must be high (e.g. > 0.95) and FAR low
             # Or simply check that it detects specific known attacks
             if metrics_canary['f1'] < 0.9: # Example threshold
                 logger.warning(f"Canary check FAILED. F1: {metrics_canary['f1']:.4f}")
                 canary_passed = False
             else:
                 logger.info("Canary check PASSED.")
        
        # Compare with Champion
        champion_path = os.path.join(self.paths['model_registry'], "champion.joblib")
        passed = False
        
        if os.path.exists(champion_path):
            champ_metrics = self._eval_single(champion_path, test_path, "Champion")
            
            f1_diff = metrics_bench['f1'] - champ_metrics['f1']
            far_diff = metrics_bench['far'] - champ_metrics['far']
            
            logger.info(f"Comparison: F1 New={metrics_bench['f1']:.4f} vs Old={champ_metrics['f1']:.4f}")
            
            # Gating Logic
            if f1_diff >= 0 and far_diff <= 0.005 and canary_passed: 
                passed = True
                if metrics_zd:
                    if metrics_zd['recall'] < 0.1:
                        logger.warning("New model failed to detect Zero-day attack completely.")
            else:
                logger.info("Model failed validation gate (Improv/FAR/Canary).")
        else:
            if canary_passed:
                logger.info("No champion found. Auto-pass (Canary safe).")
                passed = True
            else:
                logger.error("No champion found, but Canary failed. Abort.")
                passed = False
            
        return passed, {
            "benchmark": metrics_bench,
            "zeroday": metrics_zd,
            "canary": metrics_canary,
            "timestamp": self.timestamp
        }

    def _eval_single(self, model_path, data_path, tag):
        try:
            bundle = joblib.load(model_path)
            df = pd.read_csv(data_path)
            
            # Basic mapping
            feats = bundle['selected_features']
            X = df[feats]
            
            # Label
            if 'label' in df.columns:
                 # Map string label to binary if needed
                 y = df['label'].apply(lambda x: 0 if x == 'NORMAL' else 1)
            elif 'attack_type' in df.columns:
                 y = df['attack_type'].apply(lambda x: 0 if x.lower() in ['normal','benign'] else 1)
            else:
                 y = [1] * len(df) # Assumption for partial sets like ZeroDay
            
            # Ensemble Predict
            preds = []
            if bundle.get('model_lgb'): preds.append(bundle['model_lgb'].predict_proba(X)[:, 1])
            if bundle.get('model_xgb'): preds.append(bundle['model_xgb'].predict_proba(X)[:, 1])
            
            avg_prob = sum(preds) / len(preds)
            y_pred = (avg_prob >= 0.5).astype(int)
            
            f1 = f1_score(y, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
            recall = tp / (tp + fn) if (tp+fn) > 0 else 0
            far = fp / (fp + tn) if (fp+tn) > 0 else 0
            
            logger.info(f"[{tag}] F1: {f1:.4f}, Recall: {recall:.4f}, FAR: {far:.4f}")
            return {"f1": f1, "recall": recall, "far": far}
        except Exception as e:
            logger.error(f"Eval failed for {tag}: {e}")
            return {"f1": 0, "recall": 0, "far": 1}

    def deploy(self, new_model_path, metrics):
        """Step 7 & 8: Deploy & Artifacts"""
        logger.info("--- Step 7: Deployment (via Registry) ---")
        
        from utils.model_registry import ModelRegistry
        registry = ModelRegistry(self.paths['model_registry'])
        
        # Prepare artifacts map
        artifacts = {
            "classifier.joblib": new_model_path,
        }
        
        # Check for Anomaly Model (if exists from separate training)
        anomaly_path = os.path.join(self.paths['model_registry'], "anomaly_model.joblib")
        if os.path.exists(anomaly_path):
            artifacts["anomaly.joblib"] = anomaly_path
            
        # Create Version
        try:
            version_id = registry.create_version(
                artifacts=artifacts, 
                metrics=metrics, 
                config_dump=self.config
            )
            
            # Promote
            registry.promote_version(version_id)
            logger.info(f"Deployed successfully to Registry: {version_id}")
            
        except Exception as e:
            logger.error(f"Registry deployment failed: {e}")
            
    def _save_json(self, filename, data):
        path = f"logs/artifacts/{self.timestamp}"
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), "w") as f:
            json.dump(data, f, indent=4)

    def run(self, force=False):
        logger.info("=== Starting MLOps Retraining Pipeline ===")
        
        # 1. Drift Check
        if not force and not self.check_drift():
            logger.info("Pipeline stopped: No drift detected.")
            return

        # 2. Data
        train_path, zd_path = self.prepare_data()
        test_path = self.config['retraining']['evaluation']['test_set_path']
        
        # 3. Train
        new_model_path = self.train_model(train_path, test_path)
        
        # 4. Evaluate
        passed, metrics = self.evaluate(new_model_path, test_path, zd_path)
        
        # 5. Deploy
        if passed:
            self.deploy(new_model_path, metrics)
        else:
            logger.warning("Pipeline finished: Model rejected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain ignore drift")
    args = parser.parse_args()
    
    pipeline = MLOpsPipeline()
    pipeline.run(force=args.force)
