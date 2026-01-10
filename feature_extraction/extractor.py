import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .base import BaseFeatureExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

class IDSFeatureExtractor(BaseFeatureExtractor, BaseEstimator, TransformerMixin):
    """
    Feature Extractor for IDS. 
    Implements sklearn's TransformerMixin for pipeline compatibility.
    """
    def __init__(self):
        self.packet_size_stats = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn statistical parameters from training data.
        Example: Mean packet size per protocol for normalization/scaling referencing.
        """
        logger.info("Fitting Feature Extractor...")
        # Example logic: preserving consistency
        if 'protocol' in X.columns and 'length' in X.columns:
            self.packet_size_stats = X.groupby('protocol')['length'].mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering.
        Crucial: Must handle both Batch (DataFrame) and Real-time (Single Row DataFrame).
        """
        X = X.copy()
        
        # Example Feature: Protocol Frequency Encoding (Mock logic)
        # In production, use saved mappings from 'fit'
        
        # Example Feature: Byte Info
        if 'length' in X.columns:
            # Proper log transformation to handle skew and keep magnitude info
            # Add +1 to avoid log(0)
            X['length_log'] = np.log1p(X['length'].astype(float))
            # We can drop original 'length' if we want, or keep it.
            # Preprocessor identifies numeric cols. 
            # If we drop 'length', we rely on 'length_log'.
            X = X.drop(columns=['length'])
        
        return X
