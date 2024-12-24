import logging
import os
from datetime import datetime

logger = None

def setup_logger(log_dir='logs'):
    global logger
    if logger is not None:
        return logger

    logger = logging.getLogger(__name__)

    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.setLevel(logging.INFO)

        logger.info(f"Logging initialized. Log file: {log_file}")

    return logger

logger = setup_logger()
