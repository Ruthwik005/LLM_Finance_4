from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AccountState:
    cash: float
    inventory: float
    realized_pnl: float
    fees: float
    filled_bid_volume: float
    filled_ask_volume: float
    turnover: float
    _initial_wealth: float = field(default=None, repr=False)

    def __post_init__(self):
        if self._initial_wealth is None:
            self._initial_wealth = self.cash

    def apply_fill(self, side: Literal["bid", "ask"], price: float, volume: float, fee_per_unit: float) -> None:
        fee = fee_per_unit * volume
        if side == "bid":
            self.inventory += volume
            self.cash -= price * volume
            self.filled_bid_volume += volume
        else:
            self.inventory -= volume
            self.cash += price * volume
            self.filled_ask_volume += volume
        self.cash -= fee
        self.fees += fee
        self.turnover += price * volume

    def mark_to_market(self, midprice: float) -> float:
        wealth = self.cash + self.inventory * midprice
        return wealth - self._initial_wealth
