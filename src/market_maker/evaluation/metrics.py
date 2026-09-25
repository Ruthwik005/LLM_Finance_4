from __future__ import annotations
import numpy as np


def mean_absolute_position(inventory_series: np.ndarray) -> float:
    """Compute mean absolute position from inventory series."""
    return float(np.mean(np.abs(inventory_series)))


def pnl_over_map(episodic_pnl: float, map_value: float) -> float | None:
    """Compute PnL normalized by MAP.

    Returns None when map_value == 0 (never inf, NaN, or crash).
    """
    if map_value == 0:
        return None
    return episodic_pnl / map_value


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown from equity curve.

    Drawdown is the decline from a peak to a trough.
    """
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    return float(np.max(drawdowns))


def adverse_selection_ratio(fills: list[dict], midprice_series: np.ndarray, fill_indices: list[int],
                             horizon_steps: int, threshold: float) -> float:
    """Compute adverse selection ratio.

    Flags fills followed by adverse price movement beyond threshold within horizon.
    Returns ratio of adverse fills to total fills. Returns 0.0 if no fills.
    """
    if not fills:
        return 0.0
    adverse_count = 0
    for fill, idx in zip(fills, fill_indices):
        future_idx = min(idx + horizon_steps, len(midprice_series) - 1)
        move = (midprice_series[future_idx] - midprice_series[idx]) / midprice_series[idx]
        if fill["side"] == "bid" and move < -threshold:
            adverse_count += 1
        elif fill["side"] == "ask" and move > threshold:
            adverse_count += 1
    return adverse_count / len(fills)


def summarize_episode(episode_pnls: list[float], inventories: list[np.ndarray]) -> dict:
    """Summarize episode statistics across runs.

    Returns mean, median, and std of PnL across episodes.
    """
    pnls = np.array(episode_pnls, dtype=float)
    return {
        "mean_pnl": float(np.mean(pnls)),
        "median_pnl": float(np.median(pnls)),
        "std_pnl": float(np.std(pnls)),
    }
