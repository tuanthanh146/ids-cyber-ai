from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any

class BaseFeatureExtractor(ABC):
    """
    Abstract Base Class for Feature Extraction to ensure consistency
    between Training and Inference.
    """
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Any = None):
        """Fit the feature extractor parameters"""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters"""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
