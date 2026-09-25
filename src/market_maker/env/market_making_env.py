from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from market_maker.env.execution import Order, SnapshotExecutionSimulator
from market_maker.env.accounting import AccountState
from market_maker.env.reward import compute_reward
from market_maker.env.reference_price import compute_reference_price


class MarketMakingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, fused_features: np.ndarray, raw_lob_df: pd.DataFrame, feature_columns: list[str],
                 tick_size: float, levels_per_side: int, min_spread_ticks: int, max_spread_ticks: int,
                 max_mid_offset_ticks: int, total_volume_per_side: float, max_inventory: float,
                 fee_per_unit: float, reward_config: dict, reference_price_mode: str, episode_length_steps: int):
        super().__init__()
        self.fused_features = fused_features
        self.raw_lob_df = raw_lob_df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.tick_size = tick_size
        self.levels_per_side = levels_per_side
        self.min_spread_ticks = min_spread_ticks
        self.max_spread_ticks = max_spread_ticks
        self.max_mid_offset_ticks = max_mid_offset_ticks
        self.total_volume_per_side = total_volume_per_side
        self.max_inventory = max_inventory
        self.fee_per_unit = fee_per_unit
        self.reward_config = reward_config
        self.reference_price_mode = reference_price_mode
        self.episode_length_steps = episode_length_steps

        self.execution_sim = SnapshotExecutionSimulator(tick_size=tick_size)

        n_features = fused_features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._max_start = len(raw_lob_df) - episode_length_steps - 1
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._data_idx = int(self._rng.integers(0, max(1, self._max_start)))
        self.account = AccountState(cash=0.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                                     filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
        self._outstanding_orders: list[Order] = []
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        return self.fused_features[self._data_idx].astype(np.float32)

    def _current_row(self):
        return self.raw_lob_df.iloc[self._data_idx]

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        mid_offset_frac, spread_frac, bid_split_raw, ask_split_raw = action
        bid_split = (bid_split_raw + 1.0) / 2.0
        ask_split = (ask_split_raw + 1.0) / 2.0

        row = self._current_row()
        ref_price = compute_reference_price(row["bid_price_1"], row["ask_price_1"], self.reference_price_mode)

        mid_offset_ticks = mid_offset_frac * self.max_mid_offset_ticks
        spread_ticks = self.min_spread_ticks + (spread_frac + 1.0) / 2.0 * (self.max_spread_ticks - self.min_spread_ticks)
        quote_center = ref_price + mid_offset_ticks * self.tick_size
        half_spread = max(spread_ticks, 1.0) * self.tick_size / 2.0

        bid_price_l1 = self._round_to_tick(quote_center - half_spread)
        bid_price_l2 = self._round_to_tick(bid_price_l1 - self.tick_size)
        ask_price_l1 = self._round_to_tick(quote_center + half_spread)
        ask_price_l2 = self._round_to_tick(ask_price_l1 + self.tick_size)
        if bid_price_l1 >= ask_price_l1:
            ask_price_l1 = bid_price_l1 + self.tick_size
            ask_price_l2 = ask_price_l1 + self.tick_size

        bid_vol_l1 = self.total_volume_per_side * bid_split
        bid_vol_l2 = self.total_volume_per_side * (1 - bid_split)
        ask_vol_l1 = self.total_volume_per_side * ask_split
        ask_vol_l2 = self.total_volume_per_side * (1 - ask_split)

        # Clamp volumes to inventory limits (safety-critical per spec)
        remaining_long = max(0.0, self.max_inventory - self.account.inventory)
        remaining_short = max(0.0, self.max_inventory + self.account.inventory)
        bid_vol_l1 = min(bid_vol_l1, remaining_long)
        bid_vol_l2 = min(bid_vol_l2, max(0.0, remaining_long - bid_vol_l1))
        ask_vol_l1 = min(ask_vol_l1, remaining_short)
        ask_vol_l2 = min(ask_vol_l2, max(0.0, remaining_short - ask_vol_l1))

        new_orders = [
            Order(side="bid", price=bid_price_l1, volume=bid_vol_l1),
            Order(side="bid", price=bid_price_l2, volume=bid_vol_l2),
            Order(side="ask", price=ask_price_l1, volume=ask_vol_l1),
            Order(side="ask", price=ask_price_l2, volume=ask_vol_l2),
        ]
        canceled_volume = sum(o.volume for o in self._outstanding_orders)
        self._outstanding_orders = [o for o in new_orders if o.volume > 0]

        next_idx = min(self._data_idx + 1, len(self.raw_lob_df) - 1)
        next_row = self.raw_lob_df.iloc[next_idx]
        fills = self.execution_sim.step(
            self._outstanding_orders,
            next_best_bid=next_row["bid_price_1"], next_best_ask=next_row["ask_price_1"],
            next_trade_price=next_row.get("trade_price"), next_trade_side=next_row.get("trade_side"),
            next_trade_size=next_row.get("trade_size"),
        )
        for fill in fills:
            self.account.apply_fill(fill["side"], fill["price"], fill["volume"], self.fee_per_unit)

        prev_mtm = self.account.mark_to_market(ref_price)
        next_ref_price = compute_reference_price(next_row["bid_price_1"], next_row["ask_price_1"], self.reference_price_mode)
        new_mtm = self.account.mark_to_market(next_ref_price)
        delta_mtm = new_mtm - prev_mtm

        reward, components = compute_reward(
            delta_mtm_pnl=delta_mtm, inventory=self.account.inventory, canceled_volume=canceled_volume,
            reward_type=self.reward_config["type"], inventory_threshold=self.reward_config["inventory_threshold"],
            inventory_penalty=self.reward_config["inventory_penalty"],
            cancellation_penalty=self.reward_config["cancellation_penalty"], scale=self.reward_config["scale"],
        )

        self._data_idx = next_idx
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_length_steps or self._data_idx >= len(self.raw_lob_df) - 1

        obs = self._get_obs()
        info = {"fills": fills, "canceled_volume": canceled_volume, "inventory": self.account.inventory,
                "cash": self.account.cash, "reward_components": components}
        return obs, float(reward), terminated, truncated, info

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
