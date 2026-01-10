import pandas as pd
from utils.serialization import load_object
from utils.logger import get_logger

logger = get_logger(__name__)

class IDSEngine:
    """
    Production Inference Engine.
    Loads the full pipeline (Preprocessor -> FeatureExtractor -> Model).
    """
    def __init__(self, model_path: str, preprocessor_path: str, extractor_path: str):
        self.model = load_object(model_path)
        self.preprocessor = load_object(preprocessor_path)
        self.extractor = load_object(extractor_path)
        logger.info("IDSEngine initialized and artifacts loaded.")

    def predict(self, packet_data: pd.DataFrame):
        """
        End-to-end prediction for a single or batch of packets.
        """
        # 1. Feature Extraction
        features = self.extractor.transform(packet_data)
        
        # 2. Preprocessing (Scaling/Imputing)
        processed = self.preprocessor.transform(features)
        
        # 3. Prediction
        prediction = self.model.predict(processed)
        probability = self.model.predict_proba(processed)
        
        return {
            "prediction": int(prediction[0]),
            "confidence": float(probability[0].max())
        }
