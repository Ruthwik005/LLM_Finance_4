from __future__ import annotations
import math
from collections import deque
import numpy as np


class AvellanedaStoikovMarketMaker:
    """Classic Avellaneda-Stoikov (2008) quoting mapped onto the env's 4-dim action.

    reservation price  r = mid - q * gamma * sigma^2 * (T - t)
    optimal spread     delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

    Inputs:
      * mid price: the TRUE mid passed as ``predict(..., mid=)`` (``run_backtest`` supplies the env's raw mid). It is
        never recovered from standardized/clipped model features.
      * volatility: the agent keeps its own rolling window (``vol_window`` mids, config
        ``baselines.volatility_window``) of the mids it was given and uses the std of their simple returns
        (per snapshot); sigma^2 = (vol * mid)^2 in price units, (T - t) in snapshots. Call ``reset()`` at the start of
        every episode (``run_backtest`` does) so windows never straddle episodes.
      * remaining fraction (T - t)/T = obs[-4] (private state; see PRIVATE_STATE_NAMES).
    The result is converted to ticks and then to action fractions using the env's quoting grid.
    """

    def __init__(self, gamma: float, kappa: float, tick_size: float, min_spread_ticks: float,
                 max_spread_ticks: float, max_mid_offset_ticks: float, episode_length_steps: int,
                 vol_window: int = 20):
        if gamma <= 0 or kappa <= 0:
            raise ValueError("gamma and kappa must be positive")
        self.gamma, self.kappa = gamma, kappa
        self.tick_size = tick_size
        self.min_spread_ticks, self.max_spread_ticks = min_spread_ticks, max_spread_ticks
        self.max_mid_offset_ticks = max_mid_offset_ticks
        self.episode_length_steps = episode_length_steps
        self.vol_window = int(vol_window)
        self._mids: deque = deque(maxlen=self.vol_window + 1)  # vol_window returns need vol_window + 1 mids

    def reset(self) -> None:
        self._mids.clear()

    def _vol(self) -> float:
        m = np.asarray(self._mids, dtype=float)
        if len(m) < 2:
            return 0.0
        r = m[1:] / m[:-1] - 1.0
        return float(r.std(ddof=1)) if len(r) > 1 else 0.0

    def predict(self, obs: np.ndarray, inventory: float, max_inventory: float, mid: float | None = None) -> np.ndarray:
        if mid is None:
            raise ValueError("AvellanedaStoikovMarketMaker needs the true mid price: predict(..., mid=)")
        mid = float(mid)
        self._mids.append(mid)
        vol = self._vol()
        remaining_steps = float(np.clip(obs[-4], 0.0, 1.0)) * self.episode_length_steps
        var = (vol * mid) ** 2
        risk = self.gamma * var * remaining_steps
        reservation = mid - inventory * risk
        spread = risk + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        offset_ticks = (reservation - mid) / self.tick_size
        spread_ticks = spread / self.tick_size
        a0 = offset_ticks / self.max_mid_offset_ticks if self.max_mid_offset_ticks else 0.0
        span = self.max_spread_ticks - self.min_spread_ticks
        a1 = 2.0 * (spread_ticks - self.min_spread_ticks) / span - 1.0 if span else 0.0
        return np.clip(np.array([a0, a1, 0.0, 0.0], dtype=np.float32), -1.0, 1.0)
