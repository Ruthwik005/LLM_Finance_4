from __future__ import annotations
from typing import Literal


def compute_reference_price(bid_price_1: float, ask_price_1: float, mode: Literal["midprice", "imm_stable"]) -> float:
    if mode == "midprice":
        return (bid_price_1 + ask_price_1) / 2
    if mode == "imm_stable":
        # Approximation only: without message-level queue data, "stable" reference
        # falls back to midprice. This must not be presented as an exact IMM reproduction (spec 14.3).
        return (bid_price_1 + ask_price_1) / 2
    raise ValueError(f"Unknown reference_price_mode: {mode}")
