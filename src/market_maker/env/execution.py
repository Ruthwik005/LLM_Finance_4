from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Order:
    side: Literal["bid", "ask"]
    price: float
    volume: float


class SnapshotExecutionSimulator:
    """Fills are approximated from snapshot best bid/ask movement and aggregate trades,
    not a full message-level exchange reconstruction (spec section 15.3 disclaimer)."""

    def __init__(self, tick_size: float, latency_steps: int = 0):
        self.tick_size = tick_size
        self.latency_steps = latency_steps

    def step(
        self,
        outstanding_orders: list[Order],
        next_best_bid: float,
        next_best_ask: float,
        next_trade_price: Optional[float],
        next_trade_side: Optional[str],
        next_trade_size: Optional[float],
    ) -> list[dict]:
        fills = []
        for order in outstanding_orders:
            fillable_volume = 0.0

            if order.side == "bid":
                market_reached = next_best_ask <= order.price
                trade_reached = (
                    next_trade_price is not None
                    and next_trade_side == "sell"
                    and next_trade_price <= order.price
                )
                if market_reached and not trade_reached:
                    fillable_volume = order.volume
                elif trade_reached:
                    fillable_volume = min(order.volume, next_trade_size or 0.0)
                elif market_reached:
                    fillable_volume = order.volume

            elif order.side == "ask":
                market_reached = next_best_bid >= order.price
                trade_reached = (
                    next_trade_price is not None
                    and next_trade_side == "buy"
                    and next_trade_price >= order.price
                )
                if trade_reached:
                    fillable_volume = min(order.volume, next_trade_size or 0.0)
                elif market_reached:
                    fillable_volume = order.volume

            fillable_volume = min(fillable_volume, order.volume)
            if fillable_volume > 0:
                fills.append({"side": order.side, "price": order.price, "volume": fillable_volume})

        return fills
