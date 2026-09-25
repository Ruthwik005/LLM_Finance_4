"""Utility to set global random seeds for reproducibility."""
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global seed for random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
