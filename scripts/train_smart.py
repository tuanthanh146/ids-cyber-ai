import argparse
import joblib
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Setup Logging
def setup_logging(outdir):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stdout)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(outdir, "train.log"))
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def load_data(path):
    logging.info(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

def build_features(df, task, metadata_cols=None):
    if metadata_cols is None:
        metadata_cols = [
            'ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 
            'proto', 'service', 'conn_state', 'attack_type', 'label', 'risk_hint'
        ]
    
    # Identify feature columns (numeric only, excluding metadata)
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Cleaning
    X = df[numeric_cols].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True) 
    
    # Label Handling
    if task == 'multiclass':
        if 'attack_type' not in df.columns:
            logging.error("Multiclass task requested but 'attack_type' column missing.")
            sys.exit(1)
        y = df['attack_type']
        unique_labels = sorted(y.unique())
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y = y.map(label_map)
    else: # binary
        if 'label' not in df.columns:
            if 'attack_type' in df.columns:
                y = df['attack_type'].apply(lambda x: 0 if x.lower() in ['normal', 'benign'] else 1)
            else:
                logging.error("Binary task requested but 'label' column missing.")
                sys.exit(1)
        else:
            y = df['label']
        label_map = {0: 0, 1: 1}
            
    logging.info(f"Built {len(numeric_cols)} numeric features.")
    return X, y, numeric_cols, label_map

def optimize_hyperparameters(X_train, y_train, model_type, task, n_trials=20, timeout=300, seed=42):
    if not HAS_OPTUNA:
        return {}

    logging.info(f"Tuning {model_type} ({n_trials} trials, {timeout}s)...")
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)
    
    def objective(trial):
        model = None
        if model_type == 'lgbm':
            params = {
                'objective': 'binary' if task == 'binary' else 'multiclass',
                'metric': 'binary_logloss' if task == 'binary' else 'multi_logloss',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'random_state': seed,
                'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            if task == 'multiclass':
                params['num_class'] = len(np.unique(y_train))
            model = lgb.LGBMClassifier(**params)
            
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': seed,
                'n_jobs': -1
            }
            if task == 'multiclass':
                params['objective'] = 'multi:softprob'
                params['num_class'] = len(np.unique(y_train))
            
            model = xgb.XGBClassifier(**params)

        model.fit(X_t, y_t)
        y_pred = model.predict(X_v)
        
        avg_method = 'binary' if task == 'binary' else 'macro'
        return f1_score(y_v, y_pred, average=avg_method, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params

def train_model(model_type, X_train, y_train, params=None, seed=42, task='binary'):
    if params is None: params = {}
    if 'random_state' not in params: params['random_state'] = seed
    if 'n_jobs' not in params: params['n_jobs'] = -1
    
    model = None
    if model_type == 'lgbm' and HAS_LGBM:
        model = lgb.LGBMClassifier(**params, verbose=-1)
    elif model_type == 'xgb' and HAS_XGBOOST:
        if task == 'multiclass' and 'objective' not in params:
             params['objective'] = 'multi:softprob'
        model = xgb.XGBClassifier(**params)
    else:
        rf_params = {'n_estimators': 100, 'random_state': seed, 'n_jobs': -1}
        model = RandomForestClassifier(**rf_params)
    
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, y_pred, model_name, task):
    avg = 'binary' if task == 'binary' else 'macro'
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "FAR": 0.0,
        "DR": 0.0
    }
    
    if task == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["DR"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["FAR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
    return metrics

def evaluate_ensemble(models, X_test, y_test, task):
    probs = []
    for name, model in models.items():
        if model is not None:
            p = model.predict_proba(X_test)
            probs.append(p)
            
    if not probs:
        return {}, {}

    avg_prob = np.mean(probs, axis=0)
    
    if task == 'binary':
        y_pred = (avg_prob[:, 1] >= 0.5).astype(int)
    else:
        y_pred = np.argmax(avg_prob, axis=1)
        
    metrics = calculate_metrics(y_test, y_pred, "Ensemble", task)
    return metrics, avg_prob

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return pd.Series(1, index=feature_names)

def run_shap_analysis(model, X_bg, X_eval, feature_cols, outdir, task='binary'):
    if not HAS_SHAP:
        logging.warning("SHAP not installed. Skipping XAI.")
        return

    logging.info("Running SHAP XAI Analysis...")
    reports_dir = os.path.join(outdir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Init Explainer (TreeExplainer is optimized for trees)
    # Background data: summary or sample
    if len(X_bg) > 2000:
        X_bg = shap.utils.sample(X_bg, 2000)
    
    try:
        explainer = shap.TreeExplainer(model, data=X_eval if task=='binary' else None) 
        # Using X_eval for binary can handle missing values better in some versions, 
        # but often TreeExplainer works without explicit background for XGB/LGB.
        # Alternatively: explainer = shap.TreeExplainer(model)
        
        # 2. Calculate SHAP values for a subset of test data
        limit = min(2000, len(X_eval))
        shap_values = explainer.shap_values(X_eval.iloc[:limit])
        
        # Handling Binary vs Multiclass output format of shap_values
        if isinstance(shap_values, list):
            # Multiclass: shap_values is a list of arrays (one per class)
            # Binary in XGBoost might return single array, LightGBM might return list or array depending on version/objective
            # Convention: For binary, we want Class 1 (Attack). For Multi, we might want aggregate.
            
            # If binary and 2 outputs, take index 1 (Attack)
            if len(shap_values) == 2 and task == 'binary':
                vals = shap_values[1]
            else:
                # Multiclass: Aggregate absolute importance across all classes or take weighted average
                # For simplicity in this script: Sum of absolute values across classes
                vals = np.sum([np.abs(sv) for sv in shap_values], axis=0) # Shape (samples, features)
        else:
            # Single array (Binary XGBoost often returns log-odds for class 1)
            vals = shap_values

        # 3. Global Feature Importance: Mean(|SHAP|)
        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        df_imp = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)
        
        df_imp.to_csv(os.path.join(reports_dir, "shap_importance.csv"), index=False)
        logging.info("Saved reports/shap_importance.csv")

        # 4. Sample Explanations (Top 10 samples)
        # Construct JSON for dashboard
        samples = []
        for i in range(min(10, len(vals))):
            sample_shap = vals[i] # array of shape (n_features,)
            # Get Top features for this sample
            indices = np.argsort(-np.abs(sample_shap))[:10] # Top 10 features
            
            explanation = {
                "sample_index": i,
                "top_features": []
            }
            
            for idx in indices:
                feat_name = feature_cols[idx]
                feat_val = float(X_eval.iloc[i, idx])
                shap_val = float(sample_shap[idx])
                
                explanation["top_features"].append({
                    "feature": feat_name,
                    "value": feat_val,
                    "shap_value": shap_val,
                    "impact": "Positive" if shap_val > 0 else "Negative"
                })
            samples.append(explanation)
            
        with open(os.path.join(reports_dir, "shap_sample_explanations.json"), "w") as f:
            json.dump(samples, f, indent=4)
        logging.info("Saved reports/shap_sample_explanations.json")
        
    except Exception as e:
        logging.error(f"SHAP Analysis failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train Ensemble IDS (LGBM + XGB) with XAI")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--task", choices=['binary', 'multiclass'], default='binary')
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--tune", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--outdir", default="runs/ensemble")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    setup_logging(args.outdir)
    logging.info(f"Started Ensemble Training: {args}")
    
    # 1. Load Data
    df_train = load_data(args.train_csv)
    df_test = load_data(args.test_csv)
    
    X_train_full, y_train, feat_cols, label_map = build_features(df_train, args.task)
    X_test_full, y_test, _, _ = build_features(df_test, args.task)
    X_test_full = X_test_full[feat_cols]
    
    # 2. Feature Selection (Using LightGBM as proxy)
    logging.info("--- Phase 1: Feature Selection (using LightGBM) ---")
    lgb_sel = train_model('lgbm', X_train_full, y_train, seed=args.seed, task=args.task)
    importances = get_feature_importance(lgb_sel, feat_cols)
    top_k_feats = importances.head(args.top_k).index.tolist()
    logging.info(f"Selected Top {args.top_k}: {top_k_feats}")
    
    X_train = X_train_full[top_k_feats]
    X_test = X_test_full[top_k_feats]
    
    # 3. Hyperparam Tuning (if requested)
    lgbm_params = {}
    xgb_params = {}
    
    if args.tune == 1:
        logging.info("--- Phase 2: Tuning Models ---")
        if HAS_LGBM:
            lgbm_params = optimize_hyperparameters(X_train, y_train, 'lgbm', args.task, args.n_trials, args.timeout, args.seed)
        if HAS_XGBOOST:
            xgb_params = optimize_hyperparameters(X_train, y_train, 'xgb', args.task, args.n_trials, args.timeout, args.seed)
    
    # 4. Train Models
    logging.info("--- Phase 3: Training Final Models ---")
    models = {}
    metrics_list = []
    
    if HAS_LGBM:
        logging.info("Training LightGBM...")
        model_lgb = train_model('lgbm', X_train, y_train, lgbm_params, args.seed, args.task)
        metrics_lgb = calculate_metrics(y_test, model_lgb.predict(X_test), "LightGBM", args.task)
        logging.info(f"LGBM F1: {metrics_lgb['F1_Score']:.4f}")
        models['lgbm'] = model_lgb
        metrics_list.append(metrics_lgb)
        
    if HAS_XGBOOST:
        logging.info("Training XGBoost...")
        model_xgb = train_model('xgb', X_train, y_train, xgb_params, args.seed, args.task)
        metrics_xgb = calculate_metrics(y_test, model_xgb.predict(X_test), "XGBoost", args.task)
        logging.info(f"XGB F1: {metrics_xgb['F1_Score']:.4f}")
        models['xgb'] = model_xgb
        metrics_list.append(metrics_xgb)
        
    # 5. Ensemble Evaluation
    logging.info("--- Phase 4: Ensemble Evaluation ---")
    metrics_ens, _ = evaluate_ensemble(models, X_test, y_test, args.task)
    logging.info(f"Ensemble F1: {metrics_ens['F1_Score']:.4f}")
    metrics_list.append(metrics_ens)
    
    # 6. XAI (SHAP)
    logging.info("--- Phase 5: Explainability (SHAP) ---")
    # Explain the primary tree model (LightGBM preferred)
    explainer_model = models.get('lgbm') or models.get('xgb')
    if explainer_model:
        run_shap_analysis(
            explainer_model, 
            X_train, # Background data (sample)
            X_test,  # Test data to explain
            top_k_feats, 
            args.outdir,
            args.task
        )
    else:
        logging.warning("No Tree model available for SHAP.")

    # 7. Save Outputs
    logging.info("--- Phase 6: Saving Artifacts ---")
    reports_dir = os.path.join(args.outdir, "reports")
    models_dir = os.path.join(args.outdir, "models")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Compare Report
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    
    # Ensemble Bundle
    bundle = {
        "model_lgb": models.get('lgbm'),
        "model_xgb": models.get('xgb'),
        "selected_features": top_k_feats,
        "class_mapping": label_map,
        "params": {"lgbm": lgbm_params, "xgb": xgb_params}
    }
    joblib.dump(bundle, os.path.join(models_dir, "ensemble_bundle.joblib"))
    
    logging.info("Done.")

if __name__ == "__main__":
    main()
