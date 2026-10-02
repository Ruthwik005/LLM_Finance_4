from __future__ import annotations
import math
import numpy as np


class AvellanedaStoikovMarketMaker:
    """Classic Avellaneda-Stoikov (2008) quoting mapped onto the env's 4-dim action.

    reservation price  r = mid - q * gamma * sigma^2 * (T - t)
    optimal spread     delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

    Inputs are read from the observation produced by MarketMakingEnv (obs = [features | private_state]):
      * remaining fraction (T - t)/T  = obs[-4]  (private state; see PRIVATE_STATE_NAMES)
      * volatility = feature column ``rolling_vol_index`` (per-snapshot return std)
      * mid price  = feature column ``midprice_index``
    If the features were standardised, pass ``feature_mean`` / ``feature_scale`` (the scaler's
    parameters, full feature length) so the raw values are recovered. sigma^2 is expressed per
    snapshot in price units (rolling_vol * mid); (T - t) is in snapshots. The vol window itself is set
    where the feature is computed (config ``baselines.volatility_window``).
    The result is converted to ticks and then to action fractions using the env's quoting grid.
    """

    def __init__(self, gamma: float, kappa: float, rolling_vol_index: int, midprice_index: int,
                 tick_size: float, min_spread_ticks: float, max_spread_ticks: float, max_mid_offset_ticks: float,
                 episode_length_steps: int, feature_mean=None, feature_scale=None):
        if gamma <= 0 or kappa <= 0:
            raise ValueError("gamma and kappa must be positive")
        self.gamma, self.kappa = gamma, kappa
        self.vol_idx, self.mid_idx = rolling_vol_index, midprice_index
        self.tick_size = tick_size
        self.min_spread_ticks, self.max_spread_ticks = min_spread_ticks, max_spread_ticks
        self.max_mid_offset_ticks = max_mid_offset_ticks
        self.episode_length_steps = episode_length_steps
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=float)
        self.feature_scale = None if feature_scale is None else np.asarray(feature_scale, dtype=float)

    def _raw(self, obs, idx: int) -> float:
        v = float(obs[idx])
        if self.feature_scale is not None:
            v = v * self.feature_scale[idx] + (self.feature_mean[idx] if self.feature_mean is not None else 0.0)
        return v

    def predict(self, obs: np.ndarray, inventory: float, max_inventory: float) -> np.ndarray:
        mid = self._raw(obs, self.mid_idx)
        vol = max(self._raw(obs, self.vol_idx), 0.0)
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
