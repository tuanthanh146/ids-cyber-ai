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
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from utils.data_manager import DataManager
from utils.drift_detector import DriftDetector
from sklearn.metrics import f1_score, confusion_matrix

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/retrain.log")
    ]
)
logger = logging.getLogger("RetrainOrchestrator")

def load_config(path="configs/retrain_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_training_script(train_csv, test_csv, outdir, task="binary"):
    """Runs the train_smart.py script as a subprocess."""
    cmd = [
        sys.executable, "scripts/train_smart.py",
        "--train_csv", train_csv,
        "--test_csv", test_csv,
        "--task", task,
        "--outdir", outdir,
        "--tune", "1", # Auto-tune enabled for retraining
        "--n_trials", "10", # Lower trials to save time for demo, configurable
        "--top_k", "20"
    ]
    
    logger.info(f"Running training command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        raise RuntimeError("Training script failed")
    
    logger.info("Training completed successfully.")

def evaluate_model(model_bundle_path, test_csv, task='binary'):
    """Evaluates a saved model bundle against a test set."""
    if not os.path.exists(model_bundle_path):
        return None
        
    bundle = joblib.load(model_bundle_path)
    # Reconstruct ensemble
    lgbm = bundle.get("model_lgb")
    xgb = bundle.get("model_xgb")
    feats = bundle.get("selected_features")
    
    df_test = pd.read_csv(test_csv)
    
    # Preprocessing (Simplified - assuming feature extraction consistency)
    # Ideally should use the same pipeline. Here assuming test_csv is already processed features
    # We select only the features the model needs
    
    # Filter features
    try:
        X_test = df_test[feats]
    except KeyError as e:
        logger.error(f"Test set missing features required by model: {e}")
        return None
      
    # Handle labels
    if 'label' in df_test.columns:
        y_test = df_test['label']
    elif 'attack_type' in df_test.columns:
        y_test = df_test['attack_type'].apply(lambda x: 0 if x.lower() in ['normal', 'benign'] else 1)
    else:
        logger.error("No label column found in test set")
        return None
    
    # Predict
    probs = []
    if lgbm: probs.append(lgbm.predict_proba(X_test))
    if xgb: probs.append(xgb.predict_proba(X_test))
    
    if not probs:
        return None
        
    avg_prob = sum(probs) / len(probs)
    if task == 'binary':
        y_pred = (avg_prob[:, 1] >= 0.5).astype(int)
    else:
        y_pred = avg_prob.argmax(axis=1)
        
    f1 = f1_score(y_test, y_pred, average='binary' if task=='binary' else 'macro')
    
    far = 0.0
    if task == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
    return {"f1": f1, "far": far}

def main():
    parser = argparse.ArgumentParser(description="MLOps Retraining Pipeline")
    parser.add_argument("--trigger", choices=["manual", "schedule", "drift"], default="manual")
    parser.add_argument("--dry_run", action="store_true", help="Run without actual training")
    args = parser.parse_args()
    
    config = load_config()
    os.makedirs("data/storage/temp", exist_ok=True)
    os.makedirs("models/production", exist_ok=True)
    os.makedirs("models/archive", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 1. Trigger Check (Drift)
    if args.trigger == "drift":
        dm = DataManager()
        drift_detector = DriftDetector()
        
        # Load benchmark as reference and recent normal as current
        ref_df = dm.load_benchmark()
        curr_df = dm.load_recent_normal(days=3) # Check last 3 days
        
        drift_score, report = drift_detector.detect_drift(ref_df, curr_df)
        
        threshold = config['retraining']['drift_detection'].get('threshold_score', 0.5)
        logger.info(f"Drift Score: {drift_score:.4f} (Threshold: {threshold})")
        logger.info(f"Drift Report: {json.dumps(report)}") # JSON dump for clean log

        if drift_score < threshold:
            logger.info("Drift below threshold. Skipping retrain.")
            return
        else:
            logger.info("Drift Score HIGH! Proceeding to retrain.")
    
    logger.info(f"Starting Retraining Pipeline (Trigger: {args.trigger})")
    
    # 2. Data Prep
    dm = DataManager()
    train_df = dm.mix_datasets()
    if train_df.empty:
        logger.error("Mixed dataset is empty. Aborting.")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = f"data/storage/temp/train_mixed_{timestamp}.csv"
    train_df.to_csv(train_path, index=False)
    logger.info(f"Saved mixed training data to {train_path} ({len(train_df)} rows)")
    
    # For testing, we use the configured test set or a split from train if not present
    test_path = config['retraining']['evaluation']['test_set_path']
    if not os.path.exists(test_path):
        logger.warning(f"Golden test set not found at {test_path}. Using subset of training data for eval.")
        # Split train to create a temp test set
        test_df = train_df.sample(frac=0.2)
        train_df = train_df.drop(test_df.index)
        # Resave train
        train_df.to_csv(train_path, index=False)
        test_path = f"data/storage/temp/test_temp_{timestamp}.csv"
        test_df.to_csv(test_path, index=False)
        
    # 3. Train
    if args.dry_run:
        logger.info("[Dry Run] Skipping training execution.")
        return

    exp_dir = f"experiments/retrain_{timestamp}"
    try:
        run_training_script(train_path, test_path, exp_dir)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # 4. Evaluation & Gating
    new_model_path = os.path.join(exp_dir, "models", "ensemble_bundle.joblib")
    champion_path = os.path.join(config['retraining']['paths']['model_registry'], "champion.joblib")
    
    new_metrics = evaluate_model(new_model_path, test_path)
    if not new_metrics:
        logger.error("Failed to evaluate new model.")
        return
    
    logger.info(f"New Model Metrics: {new_metrics}")
    
    deploy = False
    if os.path.exists(champion_path):
        champion_metrics = evaluate_model(champion_path, test_path)
        logger.info(f"Champion Metrics: {champion_metrics}")
        
        improvement = new_metrics['f1'] - champion_metrics['f1']
        far_increase = new_metrics['far'] - champion_metrics['far']
        
        ct = config['retraining']['evaluation']
        if improvement > ct.get('min_f1_improvement', 0) and far_increase < ct.get('max_far_increase', 0.01):
            logger.info("New model outperforms Champion!")
            deploy = True
        else:
            logger.info("New model NOT better enough or FAR too high.")
    else:
        logger.info("No Champion model found. Deploying first model.")
        deploy = True
        
    # 5. Deployment
    if deploy:
        # Archive current
        if os.path.exists(champion_path):
            archive_name = f"champion_{timestamp}.joblib"
            shutil.move(champion_path, os.path.join(config['retraining']['paths']['archive_dir'], archive_name))
        
        # Promote new
        shutil.copy(new_model_path, champion_path)
        logger.info(f"Deployed new champion to {champion_path}")
        
        # Save metrics history
        with open("logs/model_history.json", "a") as f:
            log_entry = {
                "timestamp": timestamp,
                "metrics": new_metrics,
                "path": new_model_path
            }
            f.write(json.dumps(log_entry) + "\n")
            
    # Cleanup Temp
    # os.remove(train_path) 
    logger.info("Pipeline Finished.")

if __name__ == "__main__":
    main()
