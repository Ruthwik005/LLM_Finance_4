from __future__ import annotations
import numpy as np


class HeuristicMarketMaker:
    """Symmetric fixed-spread baseline with inventory-skew mid offset (spec section 20.1, Baseline A)."""

    def __init__(self, base_spread_frac: float = -0.5, skew_strength: float = 1.0):
        self.base_spread_frac = base_spread_frac
        self.skew_strength = skew_strength

    def predict(self, obs: np.ndarray, inventory: float, max_inventory: float) -> np.ndarray:
        inventory_frac = np.clip(inventory / max_inventory, -1.0, 1.0) if max_inventory else 0.0
        mid_offset = -self.skew_strength * inventory_frac  # long -> lower offset -> encourage selling
        action = np.array([mid_offset, self.base_spread_frac, 0.0, 0.0], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
