import logging
import logging.config
import yaml
import os
from pathlib import Path

def setup_logging(
    default_path='configs/logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error in Logging Configuration. Using default configs: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print('Failed to load configuration file. Using default configs')

def get_logger(name):
    return logging.getLogger(name)
