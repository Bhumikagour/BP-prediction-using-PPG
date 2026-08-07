"""
Utility functions for logging, reproducibility, random seed control,
model serialization, and timer benchmarks.
"""

import os
import sys
import time
import random
import logging
from typing import Any
import numpy as np
import joblib

def setup_logger(name: str = "BP_ML_Pipeline", log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Sets up a stream and file logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear previous handlers

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def set_seed(seed: int = 42) -> None:
    """
    Fixes random seed for Python, NumPy, and random module for 100% reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_object(obj: Any, filepath: str) -> None:
    """
    Saves a Python object (model, scaler, feature list) using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)

def load_object(filepath: str) -> Any:
    """
    Loads a Python object using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return joblib.load(filepath)

class Timer:
    """
    Context manager to time execution blocks.
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
