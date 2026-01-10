import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import argparse
import sys
# Ensure we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from utils.logger import setup_logging, get_logger
from utils.serialization import save_object
from preprocessing.preprocessor import IDSPreprocessor
from feature_extraction.extractor import IDSFeatureExtractor

logger = get_logger(__name__)

class AnomalyDetector:
    def __init__(self, output_dir="models/anomaly"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        self.threshold = None
        
    def train(self, X_train):
        """
        Train Isolation Forest ONLY on Normal data.
        """
        logger.info(f"Training IsolationForest on {X_train.shape[0]} normal samples...")
        self.model = IsolationForest(
            n_estimators=100, 
            contamination='auto', # We will manually tune threshold later
            random_state=42, 
            n_jobs=-1
        )
        self.model.fit(X_train)
        
    def find_optimal_threshold(self, X_val, y_val):
        """
        Optimize threshold based on Validation Set (Mixed Normal + Attack).
        Goal: Maximize F1 Score (Balance Precision and Recall).
        """
        logger.info(f"Tuning Threshold on {len(X_val)} samples (Attacks: {y_val.sum()})...")
        
        # decision_function: lower = more abnormal. 
        scores = self.model.decision_function(X_val)
        
        # Scan percentiles to find the best cut.
        # We look at a broad range since effective threshold depends on model separation
        percentiles = np.linspace(0.1, 50, 200) 
        thresholds = np.percentile(scores, percentiles)
        
        best_f1 = -1
        best_thresh = 0
        best_metrics = {}
        
        results = []
        
        for thresh in thresholds:
            # Predict Anomaly (1) if score < thresh
            # Convert boolean to int (1=Attack, 0=Normal)
            y_pred = (scores < thresh).astype(int)
            
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            if f1 > best_f1:
                rec = recall_score(y_val, y_pred, zero_division=0)
                prec = precision_score(y_val, y_pred, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                far = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                best_f1 = f1
                best_thresh = thresh
                best_metrics = {"recall": rec, "far": far, "f1": f1, "precision": prec}

        self.threshold = best_thresh
        logger.info(f"Optimal Threshold found: {self.threshold:.4f}")
        logger.info(f"Metrics at Opt Threshold: {best_metrics}")
        
        return scores, results
        
    def evaluate(self, X_test, y_test):
        logger.info("Evaluating on Test Set...")
        scores = self.model.decision_function(X_test)
        
        # Apply learned threshold
        y_pred = (scores < self.threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            "Detection_Rate_Recall": recall_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, zero_division=0),
            "FAR": fp / (fp + tn) if (fp+tn) > 0 else 0,
            "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }
        
        logger.info(json.dumps(metrics, indent=2))
        return scores, metrics

    def plot_scores(self, scores, y, save_name="anomaly_dist.png"):
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot Normal
            sns.histplot(scores[y==0], color='blue', label='Normal', kde=True, stat="density", bins=50, alpha=0.5)
            # Plot Anomaly
            if np.sum(y==1) > 0:
                sns.histplot(scores[y==1], color='red', label='Attack', kde=True, stat="density", bins=50, alpha=0.5)
            
            plt.axvline(self.threshold, color='k', linestyle='--', label=f'Threshold ({self.threshold:.3f})')
            plt.title('Distribution of Anomaly Scores (Lower is more Anomalous)')
            plt.xlabel('Anomaly Score (decision_function)')
            plt.legend()
            
            plt.savefig(os.path.join(self.output_dir, save_name))
            logger.info(f"Plot saved to {os.path.join(self.output_dir, save_name)}")
            plt.close()
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

def load_and_process(file_path, extractor=None, preprocessor=None, fit=False):
    """
    Load CSV, extract features (if needed), and preprocess.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
        
    logger.info(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Feature Extraction
    # If the file is raw 'processed' data (from PCAP), it might need feature extraction?
    # Our 'data/processed/normal_only.csv' seems to already have derived features like 'pkts_per_sec'?
    # Let's check headers. If it has 'ts', feature extractor might be needed to drop 'ts' etc.
    # Actually, IDSFeatureExtractor mainly adds log features.
    
    if 'label' in df.columns:
        y = df['label']
        X = df.drop(columns=['label'])
    else:
        y = None
        X = df
        
    # Drop known non-predictive cols
    drop_cols = ['ts', 'timestamp', 'src_ip', 'dst_ip', 'attack_type', 'risk_hint']
    existing_drop = [c for c in X.columns if c in drop_cols]
    if existing_drop:
        X = X.drop(columns=existing_drop)
        
    if fit:
        logger.info("Fitting Extractor and Preprocessor...")
        X_feat = extractor.fit_transform(X)
        X_proc = preprocessor.fit_transform(X_feat)
    else:
        X_feat = extractor.transform(X)
        X_proc = preprocessor.transform(X_feat)
        
    return X_proc, y

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/processed/normal_only.csv", help="Clean normal data for training")
    parser.add_argument("--test_data", default="data/processed/train.csv", help="Mixed data for threshold tuning/testing")
    parser.add_argument("--output_dir", default="models/anomaly")
    args = parser.parse_args()
    
    # Init Pipeline Components
    extractor = IDSFeatureExtractor()
    preprocessor = IDSPreprocessor()
    
    # 1. Train Data (Fit Pipeline & Model)
    try:
        X_train, _ = load_and_process(args.train_data, extractor, preprocessor, fit=True)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return

    # 2. Test/Validation Data
    # If test data is same or not provided, we might split train data or warn
    # Assuming test_data is the mixed set
    try:
        X_test_all, y_test_all = load_and_process(args.test_data, extractor, preprocessor, fit=False)
        
        # Ensure y_test_all is valid (0 and 1)
        if y_test_all is None:
            logger.warning("Test data has no labels! Cannot tune threshold properly.")
            # Mock labels? No, that's bad.
            return
            
        # Split Test Data into Validation (Tuning) and Final Test
        # Stratified split to ensure we have attacks in both
        X_val, X_test, y_val, y_test = train_test_split(X_test_all, y_test_all, test_size=0.5, random_state=42, stratify=y_test_all)
        
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        # Fallback if no test data?
        return

    # 3. Worklow
    detector = AnomalyDetector(output_dir=args.output_dir)
    
    # Train
    detector.train(X_train)
    
    # Tune
    val_scores, _ = detector.find_optimal_threshold(X_val, y_val)
    
    # Evaluate
    test_scores, metrics = detector.evaluate(X_test, y_test)
    
    # Plot
    detector.plot_scores(test_scores, y_test, save_name="anomaly_score_dist.png")
    
    # Save Artifacts
    logger.info("Saving models...")
    save_object(detector.model, os.path.join(detector.output_dir, "if_model.joblib"))
    # Save the preprocessors that were fitted on Normal data
    save_object(extractor, os.path.join(detector.output_dir, "if_extractor.joblib"))
    save_object(preprocessor, os.path.join(detector.output_dir, "if_preprocessor.joblib"))
    
    # Save threshold info
    with open(os.path.join(detector.output_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": detector.threshold, "metrics": metrics}, f)
        
    logger.info("Anomaly Detection Training Complete.")

if __name__ == "__main__":
    main()
