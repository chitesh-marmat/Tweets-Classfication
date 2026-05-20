import os
import random
import logging
import numpy as np

# Root-relative paths used everywhere in the project
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "outputs/figures"
SUBMISSIONS_DIR = "outputs/submissions"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs(*dirs):
    """Create directories if they don't exist yet."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
