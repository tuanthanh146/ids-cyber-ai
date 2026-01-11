import pandas as pd
import glob
import os
import yaml
import logging
import hashlib
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config_path="configs/retrain_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.paths = self.config["retraining"]["paths"]
        self.strategy = self.config["retraining"]["data_mixing"].get("dataset_strategy", {})
        self.max_samples = self.config["retraining"]["data_mixing"]["max_samples"]
        
        # Backward compatibility default
        if not self.strategy:
             self.strategy = {
                "sampling_ratios": {"benchmark": 0.5, "local_normal": 0.3, "local_attack": 0.2},
                "min_samples_per_attack": 50,
                "time_decay_factor": 0.0,
                "priority_weights": {"verified": 1.0, "hard_sample": 1.0}
             }

    def load_benchmark(self):
        path = self.paths["benchmark_data"]
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_recent_normal(self, days=30):
        """Load recent normal traffic stats from storage."""
        path_pattern = os.path.join(self.paths["normal_storage"], "*.csv")
        files = glob.glob(path_pattern)
        files.sort(key=os.path.getmtime, reverse=True)
        recent_files = files[:days]
        if not recent_files: return pd.DataFrame()
        
        dfs = []
        for f in recent_files:
            try:
                df = pd.read_csv(f)
                # Add source file date metadata if needed for decay
                file_ts = os.path.getmtime(f)
                df['__timestamp'] = file_ts 
                df['source_type'] = 'local_normal'
                dfs.append(df)
            except: pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def load_verified_attacks(self):
        path_pattern = os.path.join(self.paths["attack_storage"], "*.csv")
        files = glob.glob(path_pattern)
        if not files: return pd.DataFrame()
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                file_ts = os.path.getmtime(f)
                df['__timestamp'] = file_ts
                df['source_type'] = 'local_attack'
                dfs.append(df)
            except: pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _calculate_sample_weights(self, df):
        """
        Calculate sampling probability based on Time Decay and Priority.
        """
        now_ts = datetime.now().timestamp()
        decay_lambda = self.strategy.get("time_decay_factor", 0.1)
        priority_weights = self.strategy.get("priority_weights", {})
        
        weights = []
        for _, row in df.iterrows():
            w = 1.0
            
            # Time Decay
            if '__timestamp' in row:
                days_old = (now_ts - row['__timestamp']) / 86400.0
                if days_old > 0:
                    w *= np.exp(-decay_lambda * days_old)
            
            # Priority
            status = row.get('label_status', 'UNLABELED')
            if status == 'VERIFIED':
                w *= priority_weights.get("verified", 1.0)
            elif status == 'HUMAN_REVIEW_NEEDED': # Treat as hard sample
                w *= priority_weights.get("hard_sample", 1.0)
                
            weights.append(max(w, 0.001)) # Avoid zero weight
            
        return np.array(weights)

    def sanitize_retrain_data(self, df):
        """
        Apply safety filters to prevent poisoning and reduce noise.
        """
        if df.empty: return df
        
        logger.info("Sanitizing data...")
        safety_config = self.config["retraining"]["data_mixing"].get("safety", {})
        
        # 1. Provenance Check (Ensure we know where data came from)
        if 'source_type' not in df.columns:
            df['source_type'] = 'unknown' # Default
            
        # 2. Pseudo-Label Confidence Filter
        # Only apply to 'local_attack' that is NOT verified
        # Assumption: df has 'label_status', 'model_score' columns
        if 'label_status' in df.columns and 'model_score' in df.columns:
            min_conf = safety_config.get("min_confidence_pseudo_label", 0.9)
            
            # Condition: If Pseudo-Labeled (not Human Verified), Score needs to be high
            # We assume model_score is probability of attack (0-1)
            # Filter out weak pseudo-attacks
            mask_weak_pseudo = (
                (df['label_status'] == 'PSEUDO_LABELED') & 
                (df['model_score'] < min_conf)
            )
            n_dropped = mask_weak_pseudo.sum()
            if n_dropped > 0:
                logger.info(f"Dropped {n_dropped} weak pseudo-labels (conf < {min_conf})")
                df = df[~mask_weak_pseudo]

        # 3. Outlier Removal (Z-Score on key numeric features)
        # Only apply to local data (benchmark is trusted)
        local_mask = df['source_type'].isin(['local_normal', 'local_attack'])
        if local_mask.sum() > 0:
            z_thresh = safety_config.get("outlier_zscore_threshold", 4.0)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cols_to_check = [c for c in numeric_cols if c in ['duration', 'pkts_total', 'bytes_total']]
            
            for col in cols_to_check:
                # Calculate statistics only on local part
                local_data = df.loc[local_mask, col]
                if len(local_data) > 100 and local_data.std() > 0:
                     z_scores = np.abs((local_data - local_data.mean()) / local_data.std())
                     outliers = z_scores > z_thresh
                     
                     if outliers.sum() > 0:
                         # Drop outliers
                         indices_to_drop = local_data[outliers].index
                         df = df.drop(indices_to_drop)
                         local_mask = df['source_type'].isin(['local_normal', 'local_attack']) # Update mask
                         logger.info(f"Dropped {outliers.sum()} outliers in '{col}' (Z > {z_thresh})")
        
        return df

    def build_retrain_dataset(self):
        """
        Advanced mixing function with Sanitization.
        """
        logger.info("Building Optimized Retrain Dataset...")
        
        # 1. Load Data
        df_bench = self.load_benchmark()
        df_bench['source_type'] = 'benchmark'
        
        df_normal = self.load_recent_normal() # Already tags source_type
        df_attack = self.load_verified_attacks() # Already tags source_type
        
        # SANITIZATION STEP (Apply before mixing/sampling to ensure clean pool)
        df_normal = self.sanitize_retrain_data(df_normal)
        df_attack = self.sanitize_retrain_data(df_attack)

        ratios = self.strategy["sampling_ratios"]
        min_samples = self.strategy["min_samples_per_attack"]
        
        # 2. Benchmark (Simple Sampling)
        target_bench = int(self.max_samples * ratios.get("benchmark", 0.5))
        if not df_bench.empty:
            logger.info(f"Benchmark raw size: {len(df_bench)}")
            if 'label' in df_bench.columns:
                logger.info(f"Benchmark labels: {df_bench['label'].value_counts().to_dict()}")
            
            if len(df_bench) > target_bench:
                # Assuming benchmark is timeless
                final_bench = df_bench.sample(n=target_bench)
            else:
                final_bench = df_bench # Take all
        else:
            final_bench = pd.DataFrame()
            logger.warning("Benchmark DataFrame is EMPTY!")

        # 3. Local Normal (Weighted Sampling)
        target_normal = int(self.max_samples * ratios.get("local_normal", 0.3))
        final_normal = pd.DataFrame()
        if not df_normal.empty:
            weights = self._calculate_sample_weights(df_normal)
            prob = weights / weights.sum()
            # Sample
            if len(df_normal) > target_normal:
                final_normal = df_normal.sample(n=target_normal, weights=prob)
            else:
                final_normal = df_normal

        # 4. Local Attacks (Smart Sampling)
        target_attack = int(self.max_samples * ratios.get("local_attack", 0.2))
        final_attack = pd.DataFrame()
        
        if not df_attack.empty:
            # Group by Attack Type
            # Attempt to use 'final_label' or 'attack_type' or 'label'
            label_col = 'label' if 'label' in df_attack.columns else 'attack_type'
            if label_col not in df_attack.columns:
                # Fallback: treat as single blob
                weights = self._calculate_sample_weights(df_attack)
                final_attack = df_attack.sample(n=target_attack, replace=(len(df_attack)<target_attack), weights=weights/weights.sum())
            else:
                grouped = df_attack.groupby(label_col)
                temp_dfs = []
                for label, group in grouped:
                    # Enforce Min Samples
                    if len(group) < min_samples:
                        # Oversample
                        sampled = group.sample(n=min_samples, replace=True)
                    else:
                        # Calculate weights for this group
                        w = self._calculate_sample_weights(group)
                        # We don't have per-class target, but we want proportional representation roughly? 
                        # Better strategy: Sample priorities first, then fill target.
                        # For simplicity: Keep all if feasible, or downsample proportionally.
                        # Let's keep all verified attacks if possible up to target_attack cap.
                        sampled = group
                    temp_dfs.append(sampled)
                
                if temp_dfs:
                    combined_attack = pd.concat(temp_dfs)
                    # If total exceeds target, then sample down using weights
                    if len(combined_attack) > target_attack:
                        w_comb = self._calculate_sample_weights(combined_attack)
                        final_attack = combined_attack.sample(n=target_attack, weights=w_comb/w_comb.sum())
                    else:
                        final_attack = combined_attack

        # 5. Combine & Report
        final_df = pd.concat([final_bench, final_normal, final_attack], ignore_index=True)
        final_df = final_df.sample(frac=1).reset_index(drop=True)
        
        # Cleanup temp cols
        if '__timestamp' in final_df.columns: final_df.drop(columns=['__timestamp'], inplace=True)
        if 'source_type' in final_df.columns: final_df.drop(columns=['source_type'], inplace=True)

        # Hash and Report
        dhash = hashlib.sha256(pd.util.hash_pandas_object(final_df, index=False).values).hexdigest()
        
        report = {
            "total_size": len(final_df),
            "hash": dhash,
            "composition": {
                "benchmark": len(final_bench),
                "normal": len(final_normal),
                "attack": len(final_attack)
            }
        }
        if 'label' in final_df.columns:
            report['class_distribution'] = final_df['label'].value_counts().to_dict()

        logger.info(f"Dataset Built. Hash: {dhash[:8]}. Report: {report}")
        return final_df

    # Alias for backward compatibility if needed, using new logic
    def mix_datasets(self):
        return self.build_retrain_dataset()
