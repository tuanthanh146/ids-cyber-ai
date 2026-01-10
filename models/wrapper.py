import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from utils.logger import get_logger

logger = get_logger(__name__)

class IDSModel(BaseEstimator, ClassifierMixin):
    """
    Unified Wrapper for IDS Classification Models (XGBoost/LightGBM)
    """
    def __init__(self, model_type="xgboost", params=None):
        self.model_type = model_type
        self.params = params if params else {}
        self.model = None

    def fit(self, X, y):
        logger.info(f"Training {self.model_type} model...")
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**self.params)
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.model.fit(X, y)
        logger.info("Training completed.")
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
