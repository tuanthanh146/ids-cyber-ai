 import argparse
import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
from typing import Dict, Any

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import shap

from utils.logger import setup_logging, get_logger
from utils.serialization import save_object
from preprocessing.preprocessor import IDSPreprocessor
from feature_extraction.extractor import IDSFeatureExtractor

logger = get_logger(__name__)

class AdvancedTrainer:
    def __init__(self, output_dir="models/artifacts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}

    def train_baselines(self, X_train, y_train, X_test, y_test):
        logger.info("Training Baseline Models...")
        
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
            "LinearSVC": LinearSVC(class_weight='balanced', dual=False)
        }

        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self._evaluate(model, X_test, y_test, name)
            
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        logger.info("Training Advanced XGBoost...")
        
        # Calculate scale_pos_weight
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / pos
        logger.info(f"Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")

        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            enable_categorical=False, # Data is already preprocessed (one-hot)
            eval_metric='auc'
        )

        # Hyperparameter Search Space
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0]
        }

        search = RandomizedSearchCV(
            xgb_clf, 
            param_distributions=params, 
            n_iter=10, 
            scoring='f1', 
            cv=3, 
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        logger.info(f"Best XGBoost Params: {search.best_params_}")
        
        # Save Best Model
        save_object(best_model, os.path.join(self.output_dir, "model_advanced.joblib"))
        
        self._evaluate(best_model, X_test, y_test, "XGBoost_Tuned")
        
        # Feature Importance
        self._plot_feature_importance(best_model, X_train.columns)

    def _evaluate(self, model, X_test, y_test, model_name):
        preds = model.predict(X_test)
        
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, probs)
        except AttributeError:
            probs = None
            auc_score = 0.0 # LinearSVC might not have predict_proba
            if hasattr(model, "decision_function"):
                 # Approximate for SVM
                 auc_score = roc_auc_score(y_test, model.decision_function(X_test))

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics = {
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "AUC": auc_score,
            "FAR": far
        }
        
        self.results[model_name] = metrics
        logger.info(f"--- {model_name} Results ---")
        logger.info(json.dumps(metrics, indent=2))
        
    def _plot_feature_importance(self, model, feature_names):
        # 1. SHAP Importance
        try:
            # Using TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(np.array([np.zeros(len(feature_names))])) 
            # If we reached here, SHAP instantiation works.
            # Real SHAP values calculation might differ for huge datasets, but here it's fine.
            logger.info("SHAP Explainer initialized successfully.")
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")

        # 2. Built-in Importance (Robust Fallback)
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = 10
            
            logger.info("--- Top 10 Features (Built-in) ---")
            logger.info("Explanation: Higher 'Gain' implies this feature is more effective at splitting data to classify attacks.")
            for i in range(min(top_n, len(feature_names))):
                feat_name = feature_names[indices[i]]
                score = importances[indices[i]]
                logger.info(f"{i+1}. {feat_name: <20} | Score: {score:.5f}")
                
        except Exception as e:
            logger.warning(f"Could not calculate built-in feature importance: {e}")

    def save_report(self):
        report_path = os.path.join(self.output_dir, "comparison_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Comparison report saved to {report_path}")

def main():
    setup_logging()
    
    # 1. Mock Data Generation (Detailed)
    # Replacing read_csv for the standalone script
    logger.info("Generating Synthetic Data...")
    n_samples = 2000
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='S'),
        'src_ip': np.random.choice(['192.168.1.5', '192.168.1.6', '10.0.0.2', '172.16.0.1'], n_samples),
        'dst_ip': np.random.choice(['10.0.0.5', '8.8.8.8'], n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'length': np.random.randint(40, 1500, n_samples),
        'flags': np.random.choice(['SYN', 'ACK', 'FIN', 'RST', 'PSH'], n_samples),
        'ttl': np.random.randint(10, 128, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Add Target (Imbalanced)
    # Logic: High length & UDP -> higher chance of attack (mock rule)
    df['label'] = 0
    attack_mask = (df['length'] > 1000) & (df['protocol'] == 'UDP')
    df.loc[attack_mask, 'label'] = 1
    # Add random noise
    idx = np.random.choice(n_samples, size=int(n_samples*0.05), replace=False)
    df.loc[idx, 'label'] = 1
    
    logger.info(f"Data Shape: {df.shape}")
    logger.info(f"Class Distribution:\n{df['label'].value_counts()}")

    X = df.drop('label', axis=1)
    y = df['label']

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Pipeline Processing
    logger.info("Running Pipeline...")
    
    # Feature Extraction
    extractor = IDSFeatureExtractor()
    X_train_feat = extractor.fit_transform(X_train)
    X_test_feat = extractor.transform(X_test)
    
    # Preprocessing (Scaling + OneHot)
    preprocessor = IDSPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_feat)
    X_test_proc = preprocessor.transform(X_test_feat)
    
    logger.info(f"Processed Train Shape: {X_train_proc.shape}")
    
    # 4. Save Preprocessing Artifacts
    output_dir = "models/artifacts"
    save_object(extractor, os.path.join(output_dir, "extractor.joblib"))
    save_object(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    
    # 5. Advanced Training
    trainer = AdvancedTrainer(output_dir)
    trainer.train_baselines(X_train_proc, y_train, X_test_proc, y_test)
    trainer.train_xgboost(X_train_proc, y_train, X_test_proc, y_test)
    trainer.save_report()

if __name__ == "__main__":
    main()
