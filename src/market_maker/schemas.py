from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class LobRow(BaseModel):
    timestamp: datetime
    symbol: str
    bid_price_1: float
    bid_size_1: float
    ask_price_1: float
    ask_size_1: float
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    trade_side: Optional[Literal["buy", "sell", "unknown"]] = None


class NewsRow(BaseModel):
    published_at: datetime
    symbol: Optional[str] = None
    headline: str
    body: Optional[str] = None
    source: Optional[str] = None
    url_or_id: Optional[str] = None
