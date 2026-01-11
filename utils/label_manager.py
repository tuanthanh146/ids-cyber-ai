import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LabelManager:
    def __init__(self, data_path="data/storage/label_store.jsonl"):
        self.data_path = data_path
        self.whitelist_ips = ["192.168.1.100", "10.0.0.1"] # Example management IPs
        
    def load_data(self):
        """Loads labeled data from JSONL storage."""
        if not os.path.exists(self.data_path):
            return pd.DataFrame()
        
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(data)

    def save_data(self, df):
        """Saves dataframe back to JSONL."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        
        with open(self.data_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")

    def _apply_rules(self, row):
        """
        Applies heuristic rules to a single flow row.
        Returns: (label, confidence, rule_id) or (None, None, None)
        """
        # Rule 1: Whitelist
        if row.get('src_ip') in self.whitelist_ips:
            return "NORMAL", 1.0, "RULE_WHITELIST"
            
        # Rule 2: Nmap Scan Pattern (Short duration, High port)
        # Assuming features: duration, dst_port, etc. exist
        try:
            dur = float(row.get('duration', 0))
            dport = int(row.get('dst_port', 0))
            proto = str(row.get('proto', ''))
            
            if dur < 0.001 and dport > 1024 and proto == 'tcp':
                 # Heuristic: simplistic Nmap scan check
                return "PortScan", 0.9, "RULE_NMAP"
        except:
            pass
            
        # Rule 3: SYN Flood (e.g., if we had flags or pps)
        # Placeholder for SYN flood logic
        
        # Rule 4: High Confidence Model Prediction
        # If model_score is available and extreme
        score = row.get('model_score', None)
        if score is not None:
            if score < 0.001:
                return "NORMAL", 0.8, "RULE_LOW_RISK_MODEL"
            elif score > 0.999:
                return "Attack", 0.8, "RULE_HIGH_RISK_MODEL"
                
        return None, None, None

    def auto_label(self, new_flows_df):
        """
        Processes new flows and applies pseudo-labeling.
        """
        labeled_rows = []
        
        for _, row in new_flows_df.iterrows():
            item = row.to_dict()
            
            # Init metadata if new
            if 'audit_log' not in item:
                item['audit_log'] = []
                
            label, conf, rule = self._apply_rules(row)
            
            if label:
                item['label'] = label
                item['label_status'] = 'PSEUDO_LABELED'
                item['label_confidence'] = conf
                item['annotator_id'] = rule
                
                # Audit entry
                item['audit_log'].append({
                    "ts": datetime.now().isoformat(),
                    "action": "AUTO_LABEL",
                    "new_label": label,
                    "by": rule
                })
            else:
                item['label_status'] = 'UNLABELED'
                item['label'] = None
                item['label_confidence'] = 0.0
                
            labeled_rows.append(item)
            
        return pd.DataFrame(labeled_rows)

    def select_hard_samples(self, df, top_k=50):
        """
        Selects top-k samples for human review based on uncertainty and conflict.
        """
        if df.empty:
            return pd.DataFrame()
            
        # 1. Uncertainty: Score close to 0.5 (Binary)
        # Abs(score - 0.5) -> smaller is more uncertain
        df['uncertainty'] = df['model_score'].apply(lambda x: 1 - 2*abs(x - 0.5) if x is not None else 0)
        
        # 2. Rule Conflict
        # If Rule said Attack but Model Score < 0.2 OR Rule Normal but Model > 0.8
        # We need to detect this. 
        # For simplicity here, we assume if `annotator_id` is a RULE but `model_score` contradicts
        def is_conflict(row):
            lbl = row.get('label')
            score = row.get('model_score')
            if not lbl or score is None: return 0.0
            
            if lbl == 'NORMAL' and score > 0.8: return 1.0 # High conflict
            if lbl != 'NORMAL' and lbl is not None and score < 0.2: return 1.0
            return 0.0
            
        df['conflict_score'] = df.apply(is_conflict, axis=1)
        
        # Combined priority score
        # Conflict is more important than uncertainty
        df['review_priority'] = df['conflict_score'] * 2 + df['uncertainty']
        
        # Sort desc
        hard_samples = df.sort_values('review_priority', ascending=False).head(top_k)
        
        # Mark status
        if not hard_samples.empty:
            df.loc[hard_samples.index, 'label_status'] = 'HUMAN_REVIEW_NEEDED'
            
        return hard_samples

    def update_labels(self, flow_ids, new_label, annotator_id):
        """
        Updates labels for specific flows (Human verification).
        """
        df = self.load_data()
        if df.empty: return
        
        # Assuming flow_id is index or column
        timestamp = datetime.now().isoformat()
        
        for fid in flow_ids:
            # Mask
            mask = df['flow_id'] == fid
            if mask.any():
                idx = df[mask].index[0]
                old_label = df.at[idx, 'label']
                
                df.at[idx, 'label'] = new_label
                df.at[idx, 'label_status'] = 'VERIFIED'
                df.at[idx, 'label_confidence'] = 1.0
                df.at[idx, 'annotator_id'] = annotator_id
                
                # Append to audit_log (need to parse/serialize due to pandas cell limitation with list usually, 
                # but direct object assign works in memory)
                log = df.at[idx, 'audit_log']
                if isinstance(log, str): log = json.loads(log) # Safety
                if log is None: log = []
                    
                log.append({
                    "ts": timestamp,
                    "action": "MANUAL_UPDATE",
                    "old_label": old_label,
                    "new_label": new_label,
                    "by": annotator_id
                })
                df.at[idx, 'audit_log'] = log
                
        self.save_data(df)
        logger.info(f"Updated {len(flow_ids)} labels by {annotator_id}")
