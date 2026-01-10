import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

class BatchLoader:
    """
    Loads historical data for training.
    """
    def load_csv(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Loading batch data from {file_path}")
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
