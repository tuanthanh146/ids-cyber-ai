import argparse
import pandas as pd
import numpy as np
import yaml
import os
import sys
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils.logger import setup_logging, get_logger
from utils.serialization import save_object
from preprocessing.preprocessor import IDSPreprocessor
from feature_extraction.extractor import IDSFeatureExtractor
from models.trainer import IDSTrainer
from data_ingestion.batch_loader import BatchLoader

logger = get_logger(__name__)

def generate_mock_data(n_samples=2000):
    """Generates synthetic data for demonstration purposes."""
    logger.info("Generating Synthetic Data...")
    
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
    
    # Add Target (Imbalanced Mock Rule)
    # Logic: High length & UDP -> higher chance of attack
    df['label'] = 0
    attack_mask = (df['length'] > 1000) & (df['protocol'] == 'UDP')
    df.loc[attack_mask, 'label'] = 1
    
    # Add random noise
    idx = np.random.choice(n_samples, size=int(n_samples*0.05), replace=False)
    df.loc[idx, 'label'] = 1
    
    return df

def train_pipeline(config_path, data_path=None, output_dir=None, demo_mode=False):
    setup_logging()
    
    # Load Config
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Determine Output Directory
    if output_dir is None:
        output_dir = config['paths']['models']
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Artifacts will be saved to: {output_dir}")

    # 1. Load Data
    X = None
    if demo_mode:
        logger.info("Running in DEMO mode.")
        X = generate_mock_data()
    elif data_path:
        logger.info(f"Loading data from {data_path}")
        loader = BatchLoader()
        X = loader.load_csv(data_path)
        if X.empty:
            logger.error("Loaded data is empty. Exiting.")
            return
    else:
        logger.error("No data provided. Use --data <path> or --demo")
        return

    # Check for target column
    target_col = config['model'].get('target_col', 'label')
    if target_col not in X.columns:
        logger.error(f"Target column '{target_col}' not found in data.")
        return

    y = X.pop(target_col)
    
    # 2. Split Data
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        logger.error(f"Training data must contain at least 2 classes (0 and 1). Found only: {unique_classes}")
        logger.error("Please provide a dataset with both benign and attack traffic.")
        return

    logger.info("Splitting data...")
    # Exclude non-predictive or high-cardinality columns
    drop_cols = ['ts', 'timestamp', 'attack_type', 'src_ip', 'dst_ip', 'risk_hint'] # IPs can be removed to prevent overfitting to specific addresses
    existing_drop = [c for c in X.columns if c in drop_cols]
    if existing_drop:
        logger.info(f"Dropping columns to prevent overfitting/leakage: {existing_drop}")
        X = X.drop(columns=existing_drop)

    # Stratify if classification to maintain class balance
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if stratify fails (e.g. usage of regression or too few samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # 3. Feature Extraction
    logger.info("Running Feature Extraction...")
    extractor = IDSFeatureExtractor()
    # Fit on train, transform on train
    X_train_extracted = extractor.fit_transform(X_train)
    # Transform on test
    X_test_extracted = extractor.transform(X_test)
    
    # 4. Preprocessing
    logger.info("Running Preprocessing...")
    preprocessor = IDSPreprocessor()
    # Fit on train, transform on train
    X_train_processed = preprocessor.fit_transform(X_train_extracted)
    # Transform on test
    X_test_processed = preprocessor.transform(X_test_extracted)
    
    logger.info(f"Training Data Shape: {X_train_processed.shape}")
    logger.info(f"Test Data Shape: {X_test_processed.shape}")

    # 5. Model Training
    logger.info(f"Training Model ({config['model']['type']})...")
    trainer = IDSTrainer(config)
    model = trainer.train(X_train_processed, y_train, X_test_processed, y_test)
    
    # 6. Save Artifacts
    logger.info("Saving Artifacts...")
    
    # Save Model
    save_object(model, os.path.join(output_dir, "model.joblib"))
    
    # Save Pipelines
    save_object(extractor, os.path.join(output_dir, "extractor.joblib"))
    save_object(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    
    # Save specific evaluation metrics if needed
    # (Trainer logs report, but we could save a json report here too)
    
    logger.info("Training pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS Training Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--data", help="Path to training data (CSV)")
    parser.add_argument("--output", help="Directory to save artifacts")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic data")
    
    args = parser.parse_args()
    
    train_pipeline(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        demo_mode=args.demo
    )
