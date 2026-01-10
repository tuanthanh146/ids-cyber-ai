import pytest
from models.fusion import DecisionFusion

class TestDecisionFusion:
    def setup_method(self):
        self.fusion = DecisionFusion(anomaly_threshold=-0.12)
        
    def test_normal_traffic(self):
        # XGB: Normal (Conf 0.9 => Attack Conf 0.1), Anomaly: Normal (Score 0.2)
        res = self.fusion.fuse(xgb_label=0, xgb_proba=0.1, anomaly_score=0.2)
        assert res['final_label'] == 0
        assert res['alert_level'] == "LOW"
        
    def test_known_attack_high_conf(self):
        # XGB: Attack (0.95), Anomaly: Maybe (Score -0.13)
        res = self.fusion.fuse(xgb_label=1, xgb_proba=0.95, anomaly_score=-0.13)
        assert res['final_label'] == 1
        assert res['risk_score'] > 50
        assert "Known Signature Match" in res['reasons']
        
    def test_zero_day_attack(self):
        # XGB: Normal (Missed it), Anomaly: Strong (Score -0.3)
        res = self.fusion.fuse(xgb_label=0, xgb_proba=0.2, anomaly_score=-0.3)
        assert res['final_label'] == 1
        assert res['alert_level'] in ["HIGH", "CRITICAL"]
        assert "Statistical Anomaly" in res['reasons']
        assert "Potential Zero-day" in res['reasons']
        
    def test_weak_signal_fusion(self):
        # XGB: Attack but low conf (0.55), Anomaly: Normal
        res = self.fusion.fuse(xgb_label=1, xgb_proba=0.55, anomaly_score=0.1)
        # Score = 0.55 * 60 = 33. Anomaly = 0. Total = 33.
        # Should be Medium risk, maybe Label 1
        assert res['risk_score'] >= 30
        assert res['alert_level'] == "MEDIUM" 
