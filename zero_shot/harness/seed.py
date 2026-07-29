# -*- coding: utf-8 -*-
"""
seed.py - Deterministic seed enforcement across all random sources.
"""

import os
import random

import numpy as np
import torch

from shared.corruption import stable_corruption_seed

from .config import GLOBAL_SEED


__all__ = ["set_global_seed", "stable_corruption_seed"]


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Enforce deterministic execution globally."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
