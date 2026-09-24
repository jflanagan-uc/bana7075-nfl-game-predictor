import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import LOGS_PATH, PROCESSED_DATA_PATH, MODELS_PATH

for _p in (LOGS_PATH, PROCESSED_DATA_PATH, MODELS_PATH):
    Path(_p).mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        fh = logging.FileHandler(os.path.join(LOGS_PATH, f"{name}.log"), encoding="utf-8")
        ch = logging.StreamHandler()
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def safe_div(n, d):
    """Element-wise n / d that returns NaN (not inf) when d is 0 or missing."""
    n = pd.to_numeric(n, errors="coerce").astype(float)
    d = pd.to_numeric(d, errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = n / d
    return out.where(d != 0)
