from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from market_maker.env.execution import Order, SnapshotExecutionSimulator
from market_maker.env.accounting import AccountState
from market_maker.env.reward import compute_reward
from market_maker.env.reference_price import compute_reference_price


# Private-state layout appended to every observation: obs = [market_features | private_state].
# Indices relative to the start of the private block (also obs[-6:]):
#   0 inventory / max_inventory, clipped to [-1, 1]
#   1 tanh(mark_to_market_pnl / pnl_scale)
#   2 remaining episode fraction in [0, 1]  (1 at reset, 0 at the end)
#   3 last quoted L1 bid distance from the reference price, in ticks / max_spread_ticks
#   4 last quoted L1 ask distance from the reference price, in ticks / max_spread_ticks
#   5 last quoted L1 spread (ask_l1 - bid_l1), in ticks / max_spread_ticks
PRIVATE_STATE_DIM = 6
PRIVATE_STATE_NAMES = ["inventory_frac", "mtm_pnl_tanh", "remaining_frac",
                       "last_bid_distance_norm", "last_ask_distance_norm", "last_spread_norm"]


class MarketMakingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, fused_features: np.ndarray, raw_lob_df: pd.DataFrame, feature_columns: list[str],
                 tick_size: float, levels_per_side: int, min_spread_ticks: int, max_spread_ticks: int,
                 max_mid_offset_ticks: int, total_volume_per_side: float, max_inventory: float,
                 fee_per_unit: float, reward_config: dict, reference_price_mode: str, episode_length_steps: int,
                 rebate_per_unit: float = 0.0, latency_steps: int = 0, pnl_scale: float = 50.0):
        super().__init__()
        self.fused_features = fused_features
        self.raw_lob_df = raw_lob_df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.tick_size = tick_size
        self.levels_per_side = levels_per_side
        if levels_per_side != 2:
            raise ValueError(f"MarketMakingEnv currently only supports levels_per_side=2, got {levels_per_side}")
        self.min_spread_ticks = min_spread_ticks
        self.max_spread_ticks = max_spread_ticks
        self.max_mid_offset_ticks = max_mid_offset_ticks
        self.total_volume_per_side = total_volume_per_side
        self.max_inventory = max_inventory
        self.fee_per_unit = fee_per_unit
        self.rebate_per_unit = rebate_per_unit
        self.latency_steps = int(latency_steps)
        self.pnl_scale = float(pnl_scale)
        self.reward_config = reward_config
        self.reference_price_mode = reference_price_mode
        self.episode_length_steps = episode_length_steps

        self.execution_sim = SnapshotExecutionSimulator(tick_size=tick_size, latency_steps=self.latency_steps)

        n_features = fused_features.shape[1]
        # market features unbounded; private block: inv [-1,1], mtm tanh [-1,1], remaining [0,1],
        # bid/ask distance [-1,1], spread [0,1]
        low = np.concatenate([np.full(n_features, -np.inf), [-1.0, -1.0, 0.0, -1.0, -1.0, 0.0]]).astype(np.float32)
        high = np.concatenate([np.full(n_features, np.inf), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._max_start = len(raw_lob_df) - episode_length_steps - 1
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        start = (options or {}).get("start_index")
        if start is None:
            self._data_idx = int(self._rng.integers(0, max(1, self._max_start)))
        else:
            # Deterministic window start (validation/test). Valid range keeps a full episode inside the data.
            start = int(start)
            if not 0 <= start <= max(0, self._max_start):
                raise ValueError(f"start_index {start} outside [0, {max(0, self._max_start)}]")
            self._data_idx = start
        self.account = AccountState(cash=0.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                                     filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
        # Outstanding resting orders keyed by (side, level); value = (Order with remaining volume, placed_step)
        self._outstanding_orders: dict[tuple[str, int], tuple[Order, int]] = {}
        self._last_quote_norm = (0.0, 0.0, 0.0)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        row = self._current_row()
        ref = compute_reference_price(row["bid_price_1"], row["ask_price_1"], self.reference_price_mode)
        inv_frac = float(np.clip(self.account.inventory / self.max_inventory, -1.0, 1.0)) if self.max_inventory else 0.0
        mtm = float(np.tanh(self.account.mark_to_market(ref) / self.pnl_scale))
        remaining = max(0.0, 1.0 - self._step_idx / self.episode_length_steps)
        bd, ad, sp = self._last_quote_norm
        private = np.array([inv_frac, mtm, remaining, bd, ad, sp], dtype=np.float32)
        return np.concatenate([self.fused_features[self._data_idx].astype(np.float32), private])

    def _current_row(self):
        return self.raw_lob_df.iloc[self._data_idx]

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        mid_offset_frac, spread_frac, bid_split_raw, ask_split_raw = action
        bid_split = (bid_split_raw + 1.0) / 2.0
        ask_split = (ask_split_raw + 1.0) / 2.0

        row = self._current_row()
        ref_price = compute_reference_price(row["bid_price_1"], row["ask_price_1"], self.reference_price_mode)
        prev_mtm = self.account.mark_to_market(ref_price)

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

        desired = {
            ("bid", 0): Order(side="bid", price=bid_price_l1, volume=bid_vol_l1),
            ("bid", 1): Order(side="bid", price=bid_price_l2, volume=bid_vol_l2),
            ("ask", 0): Order(side="ask", price=ask_price_l1, volume=ask_vol_l1),
            ("ask", 1): Order(side="ask", price=ask_price_l2, volume=ask_vol_l2),
        }
        # Quote persistence (spec 15.2): an unchanged price keeps its remaining volume (and queue/latency state).
        #  - desired > remaining: the slot is topped up to the desired volume; the difference is counted as
        #    placed_volume in the step log (top-ups are treated as plain placed volume, no queue-priority model).
        #  - desired < remaining: the difference is canceled (counted as canceled_volume); the rest keeps priority.
        #  - changed price: the old remaining volume is canceled and the new volume placed (restarts latency).
        # Desired volumes are already clamped so that per-side total <= max_inventory -/+ inventory, hence even if
        # every outstanding order filled, |inventory| <= max_inventory.
        placed_volume = 0.0
        canceled_volume = 0.0
        reconciled: dict[tuple[str, int], tuple[Order, int]] = {}
        for key, new in desired.items():
            old = self._outstanding_orders.get(key)
            if old is not None:
                old_order, old_step = old
                if abs(old_order.price - new.price) < 1e-12 and new.volume > 0:
                    diff = new.volume - old_order.volume
                    if diff > 1e-12:
                        placed_volume += diff
                        old_order.volume = new.volume
                    elif diff < -1e-12:
                        canceled_volume += -diff
                        old_order.volume = new.volume
                    reconciled[key] = (old_order, old_step)
                    continue
                canceled_volume += old_order.volume
            if new.volume > 0:
                placed_volume += new.volume
                reconciled[key] = (Order(side=new.side, price=new.price, volume=new.volume), self._step_idx)
        self._outstanding_orders = reconciled
        # Quoted state for the log: prices always shown, volumes = outstanding remaining volume (0 if none)
        quoted_volumes = {k: (reconciled[k][0].volume if k in reconciled else 0.0) for k in desired}

        if bid_vol_l1 > 0 or bid_vol_l2 > 0 or ask_vol_l1 > 0 or ask_vol_l2 > 0:
            self._last_quote_norm = (
                float(np.clip((ref_price - bid_price_l1) / self.tick_size / self.max_spread_ticks, -1.0, 1.0)),
                float(np.clip((ask_price_l1 - ref_price) / self.tick_size / self.max_spread_ticks, -1.0, 1.0)),
                float(np.clip((ask_price_l1 - bid_price_l1) / self.tick_size / self.max_spread_ticks, 0.0, 1.0)),
            )

        next_idx = min(self._data_idx + 1, len(self.raw_lob_df) - 1)
        next_row = self.raw_lob_df.iloc[next_idx]
        eligible_keys = [k for k, (_, placed_step) in self._outstanding_orders.items()
                         if self._step_idx >= placed_step + self.latency_steps]
        eligible = [self._outstanding_orders[k][0] for k in eligible_keys]
        fills = self.execution_sim.step(
            eligible,
            next_best_bid=next_row["bid_price_1"], next_best_ask=next_row["ask_price_1"],
            next_trade_price=next_row.get("trade_price"), next_trade_side=next_row.get("trade_side"),
            next_trade_size=next_row.get("trade_size"),
        )
        next_ref_price = compute_reference_price(next_row["bid_price_1"], next_row["ask_price_1"], self.reference_price_mode)
        fills = [dict(f, mid=float(next_ref_price)) for f in fills]
        fees_before, rebates_before = self.account.fees, self.account.rebates
        for fill in fills:
            self.account.apply_fill(fill["side"], fill["price"], fill["volume"], self.fee_per_unit,
                                    self.rebate_per_unit)
            # partial fills reduce the resting remainder; the unfilled remainder persists
            for k in eligible_keys:
                o = self._outstanding_orders[k][0]
                if o.side == fill["side"] and abs(o.price - fill["price"]) < 1e-12:
                    o.volume -= fill["volume"]
                    break
        step_fees = self.account.fees - fees_before
        step_rebates = self.account.rebates - rebates_before
        self._outstanding_orders = {k: v for k, v in self._outstanding_orders.items() if v[0].volume > 1e-12}
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
        ts = row.get("timestamp") if hasattr(row, "get") else None
        step_log = {
            "t": self._step_idx - 1,
            "timestamp": None if ts is None else pd.Timestamp(ts).isoformat(),
            "mid": float(ref_price),
            "next_mid": float(next_ref_price),
            "bid_prices": [float(bid_price_l1), float(bid_price_l2)],
            "ask_prices": [float(ask_price_l1), float(ask_price_l2)],
            "bid_volumes": [float(quoted_volumes[("bid", 0)]), float(quoted_volumes[("bid", 1)])],
            "ask_volumes": [float(quoted_volumes[("ask", 0)]), float(quoted_volumes[("ask", 1)])],
            "inventory": float(self.account.inventory),
            "cash": float(self.account.cash),
            "mtm_pnl": float(new_mtm),
            "fills": [dict(f) for f in fills],
            "placed_volume": float(placed_volume),
            "canceled_volume": float(canceled_volume),
            "fees_paid": float(step_fees),
            "rebates_received": float(step_rebates),
        }
        info = {"fills": fills, "canceled_volume": canceled_volume, "placed_volume": placed_volume,
                "inventory": self.account.inventory, "cash": self.account.cash,
                "fees_paid": self.account.fees, "rebates_received": self.account.rebates,
                "reward_components": components, "step_log": step_log}
        return obs, float(reward), terminated, truncated, info

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
