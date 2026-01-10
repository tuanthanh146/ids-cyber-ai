from utils.logger import get_logger
from .wrapper import IDSModel
from sklearn.metrics import classification_report

logger = get_logger(__name__)

class IDSTrainer:
    def __init__(self, config):
        self.config = config
        
    def train(self, X_train, y_train, X_test, y_test):
        logger.info("Starting training pipeline...")
        
        # Initialize Model
        model_params = self.config.get('model_params', {})
        clf = IDSModel(model_type=self.config['model']['type'], params=model_params)
        
        # Fit
        clf.fit(X_train, y_train)
        
        # Evaluate
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds)
        logger.info(f"Evaluation Results:\n{report}")
        
        return clf
