import joblib
import os
from typing import Any
from .logger import get_logger

logger = get_logger(__name__)

def save_object(obj: Any, file_path: str):
    """Save object to disk using joblib"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {e}")
        raise e

def load_object(file_path: str) -> Any:
    """Load object from disk using joblib"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        obj = joblib.load(file_path)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {e}")
        raise e
