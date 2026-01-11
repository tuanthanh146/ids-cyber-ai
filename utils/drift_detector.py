import pandas as pd
import numpy as np
from scipy import stats
import yaml
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, config_path="configs/retrain_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.dd_config = self.config["retraining"]["drift_detection"]
        self.monitored_features = self.dd_config["monitored_features"]
        self.retrain_thresh = self.dd_config.get("retrain_threshold", 0.8)
        self.weights = self.dd_config.get("weights", {"feature_psi": 0.6, "prediction_drift": 0.4})
        
    def calculate_psi(self, expected, actual, buckettype='bins', buckets=10, axis=0):
        """Calculate the PSI (Population Stability Index) for a single feature"""
        try:
            expected = expected.dropna()
            actual = actual.dropna()
            
            if len(expected) == 0 or len(actual) == 0:
                return 0.0

            def scale_range(input, min, max):
                input += -(np.min(input))
                input /= np.max(input) / (max - min)
                input += min
                return input

            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

            if buckettype == 'bins':
                breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
            elif buckettype == 'quantiles':
                breakpoints = np.stack([np.percentile(expected, b) for b in breakpoints])

            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

            def sub_psi(e_perc, a_perc):
                if a_perc == 0: a_perc = 0.0001
                if e_perc == 0: e_perc = 0.0001
                return (e_perc - a_perc) * np.log(e_perc / a_perc)

            psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))
            return psi_value
        except Exception as e:
            logger.error(f"PSI Error: {e}")
            return 0.0

    def check_prediction_drift(self, alert_logs_path="logs/alerts.jsonl"):
        """
        Check if the Alert Rate implies drift (e.g. sudden spike in HIGH alerts).
        Returns a score [0, 1] magnitude of drift.
        """
        if not os.path.exists(alert_logs_path):
            return 0.0
            
        try:
            # Load last 1000 alerts
            # Improving performance: read from end? For now standard read
            df = pd.read_json(alert_logs_path, lines=True)
            if df.empty: return 0.0
            
            # Convert timestamp
            df['dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce').fillna(pd.to_datetime(df['timestamp'], errors='coerce'))
            df = df.sort_values('dt')
            
            # Split into Reference (First 50%) and Recent (Last 20%) ??? 
            # Or Reference = last week vs Recent = today.
            # Simplified: Reference = All previous, Recent = Last 24h
            recent_cutoff = df['dt'].max() - pd.Timedelta(days=1)
            
            recent_df = df[df['dt'] >= recent_cutoff]
            ref_df = df[df['dt'] < recent_cutoff]
            
            if ref_df.empty or recent_df.empty:
                return 0.0
                
            # Metric: % of HIGH/CRITICAL alerts
            def get_severe_rate(d):
                return len(d[d['alert_level'].isin(['HIGH', 'CRITICAL'])]) / len(d)
                
            ref_rate = get_severe_rate(ref_df)
            rec_rate = get_severe_rate(recent_df)
            
            delta = abs(rec_rate - ref_rate)
            
            # Norm logic: if delta > max_allowed (0.1), score -> 1.0
            max_allowed = self.dd_config.get("prediction_drift", {}).get("max_alert_rate_change", 0.1)
            
            drift_score = min(delta / max_allowed, 1.0)
            return drift_score
            
        except Exception as e:
            logger.error(f"Prediction Drift Check Failed: {e}")
            return 0.0

    def detect_drift(self, reference_df, current_df):
        """
        Aggregated Drift Detection.
        Returns:
            drift_score (float): 0.0 to 1.0
            report (dict): Detailed breakdown
        """
        report = {}
        psi_values = []
        
        # 1. Feature Drift (PSI)
        common_features = [f for f in self.monitored_features if f in reference_df.columns and f in current_df.columns]
        
        for feature in common_features:
            psi = self.calculate_psi(reference_df[feature], current_df[feature], buckettype='quantiles', buckets=10)
            psi_values.append(psi)
            report[feature] = {"psi": float(psi), "drift": psi > 0.2}
            
        avg_psi = np.mean(psi_values) if psi_values else 0.0
        
        # Normalize PSI to [0,1] for score. 
        # Heuristic: PSI=0.2 is bad -> score=0.5? PSI=0.5 -> score=1.0?
        # Let's say score = Clip(PSI / 0.4, 1.0)
        feat_score = min(avg_psi / 0.4, 1.0)
        report["avg_psi"] = float(avg_psi)
        report["feature_score"] = float(feat_score)

        # 2. Prediction Drift
        pred_score = self.check_prediction_drift()
        report["prediction_score"] = float(pred_score)
        
        # 3. Aggregation
        w_f = self.weights["feature_psi"]
        w_p = self.weights["prediction_drift"]
        
        final_score = (feat_score * w_f) + (pred_score * w_p)
        report["final_score"] = float(final_score)
        
        is_drift = final_score > self.config["retraining"]["drift_detection"].get("threshold_score", 0.5)
        should_retrain = final_score > self.retrain_threshold
        
        report["status"] = "RETRAIN_NEEDED" if should_retrain else ("WARNING" if is_drift else "STABLE")
        
        return final_score, report
