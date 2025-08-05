# utils/logger.py

import logging
import os

def setup_logger(name="spam_classifier", log_file="logs/app.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    logger = logging.getLogger(name)
    if logger.handlers: 
        return logger

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.setLevel(level)
    return logger