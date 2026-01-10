import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from utils.logger import get_logger

logger = get_logger(__name__)

class IDSPreprocessor:
    """
    Wrapper for Scikit-Learn Pipeline to handle Clean + Scale
    """
    def __init__(self):
        self.pipeline = None
        self.numeric_features = []
        self.categorical_features = []

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting Preprocessor...")
        
        # Identify Column Types
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numeric Features: {self.numeric_features}")
        logger.info(f"Categorical Features: {self.categorical_features}")

        # Create Transformer
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        self.pipeline.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
            
        array_output = self.pipeline.transform(X)
        
        # Get feature names if possible
        try:
            feature_names = self.pipeline.get_feature_names_out()
        except AttributeError:
            # Fallback if scikit-learn version is old or other issue
            feature_names = [f"feat_{i}" for i in range(array_output.shape[1])]
            
        return pd.DataFrame(array_output, columns=feature_names)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
