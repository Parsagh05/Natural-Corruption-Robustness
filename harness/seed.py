# -*- coding: utf-8 -*-
"""
seed.py - Deterministic seed enforcement across all random sources.
"""

import os
import random
import hashlib

import numpy as np
import torch

from .config import GLOBAL_SEED


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Enforce deterministic execution globally."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_corruption_seed(base_seed: int, rel_path: str, corruption: str, severity: int) -> int:
    """
    Generate a deterministic per-sample seed for corruption reproducibility.
    Mirrors the logic in natural_noise.py.
    """
    key = f"{base_seed}|{rel_path}|{corruption}|{severity}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
