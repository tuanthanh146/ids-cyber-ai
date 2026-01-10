import numpy as np
from typing import Dict, Tuple, Any

class DecisionFusion:
    """
    Implements Decision Fusion logic to combine Supervised (XGBoost) 
    and Unsupervised (Isolation Forest) models.
    
    Strategy: Rule-based Heuristic with Risk Scoring.
    """
    def __init__(self, anomaly_threshold: float = -0.12):
        self.anomaly_threshold = anomaly_threshold
        
    def fuse(self, xgb_label: int, xgb_proba: float, anomaly_score: float) -> Dict[str, Any]:
        """
        Fuse outputs to determine final alert level and risk score.
        
        Args:
            xgb_label: 0 (Normal) or 1 (Attack)
            xgb_proba: Confidence of XGBoost (0.0 to 1.0)
            anomaly_score: Raw score from IsolationForest (lower = more anomalous)
            
        Returns:
            Dict containing:
            - final_label: 0 or 1
            - risk_score: 0.0 to 100.0
            - alert_level: LOW, MEDIUM, HIGH, CRITICAL
            - reason: Explanation string
        """
        
        # Normalize Anomaly Score to a Risk Factor (Approximate)
        # IF Score usually -0.5 (Very Anom) to +0.5 (Very Normal)
        # We want to map this to 0 (Normal) -> 1 (Anomaly) roughly for calculation
        # Heuristic: 
        # If score < threshold, it is definitely an anomaly.
        # We use sigmoid-like scaling or simple linear clip.
        
        is_anomaly = anomaly_score < self.anomaly_threshold
        
        # Calculate Base Risk Score (0-100)
        risk_score = 0.0
        reason = []
        
        # 1. XGBoost Contribution
        if xgb_label == 1:
            risk_score += xgb_proba * 60  # Max 60 points from XGB
            reason.append(f"Known Signature Match ({xgb_proba:.2f})")
        
        # 2. Anomaly Contribution
        # If score is very low (e.g., -0.2), risk increases significantly
        # If score is near threshold, risk is moderate
        # Scaling: inverted score.
        if is_anomaly:
            # Distance from threshold
            # Multiplier 200: 0.1 diff = 20 points
            severity = abs(anomaly_score - self.anomaly_threshold) * 200 
            # Cap anomaly contribution
            # Base 30 for just being anomaly + severity
            anomaly_points = min(70, severity + 30) 
            risk_score += anomaly_points
            reason.append(f"Statistical Anomaly (Score {anomaly_score:.2f})")
        
        # Cap final score
        risk_score = min(100.0, risk_score)
        
        # Determine Final Label and Level
        final_label = 0
        alert_level = "LOW"
        
        if risk_score >= 80:
            final_label = 1
            alert_level = "CRITICAL"
        elif risk_score >= 50:
            final_label = 1
            alert_level = "HIGH"
        elif risk_score >= 30:
            # Could be a false alarm or weak anomaly
            # If XGB said 1 but weak confidence?
            final_label = 1 if xgb_label == 1 else 0 
            alert_level = "MEDIUM"
        else:
            final_label = 0
            alert_level = "LOW"
            
        # Special Case: Zero-day (XGB Normal, but High Anomaly)
        if xgb_label == 0 and is_anomaly and risk_score > 40:
            final_label = 1
            alert_level = "HIGH"
            reason.append("Potential Zero-day")

        return {
            "final_label": final_label,
            "risk_score": round(risk_score, 2),
            "alert_level": alert_level,
            "reasons": ", ".join(reason) if reason else "Normal Traffic"
        }
