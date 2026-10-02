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


# ---------------------------------------------------------------------------
# Trader-grade metrics computed from per-step logs (env info["step_log"]).
# ---------------------------------------------------------------------------

def _cfg_get(config, section: str) -> dict:
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    return dict(config.get(section, {}) or {})


def adverse_selection_from_steps(steps: list[list[dict]], horizon_steps: int, threshold_price: float) -> float | None:
    """Fraction of fills whose mid moved against them by MORE than threshold_price within horizon_steps.

    A fill at step t happens at mid m = steps[t]["next_mid"]; the comparison mid is steps[t + h]["next_mid"].
    Buys are adverse if future - m < -threshold, sells if future - m > threshold. Fills without a full
    horizon remaining in the episode are not evaluable and are skipped. Returns None if none are evaluable.
    """
    adverse = total = 0
    for ep in steps:
        for t, s in enumerate(ep):
            if t + horizon_steps >= len(ep):
                continue
            future = ep[t + horizon_steps]["next_mid"]
            for f in s["fills"]:
                total += 1
                move = future - f.get("mid", s["next_mid"])
                if (f["side"] == "bid" and move < -threshold_price) or (f["side"] == "ask" and move > threshold_price):
                    adverse += 1
    return adverse / total if total else None


def summarize_run(backtest_result: dict, config) -> dict:
    """Flat metrics dict (floats only). Undefined metrics are omitted, never inf/NaN.

    PnL attribution (totals over all evaluated episodes):
      spread_capture_pnl = sum buys vol*(mid_at_fill - price) + sells vol*(price - mid_at_fill)
      inventory_pnl      = total_pnl - spread_capture_pnl + fees_paid - rebates_received
    (fees reduce and rebates increase total_pnl, so they are removed to isolate price-drift PnL.)
    Distances/spreads are in ticks. max_drawdown is the worst intra-episode drawdown of the mtm equity curve.
    """
    cfg = config.model_dump() if hasattr(config, "model_dump") else config
    sim, ev = cfg.get("simulator", {}), _cfg_get(cfg, "evaluation")
    tick = cfg["data"]["tick_size"]
    horizon = int(ev.get("adverse_horizon_steps", 10))
    thr_ticks = float(ev.get("adverse_threshold_ticks", 1.0))
    fee, rebate = float(sim.get("fee_per_unit", 0.0)), float(sim.get("rebate_per_unit", 0.0))

    pnls = list(backtest_result["episode_pnls"])
    invs = backtest_result["inventories"]
    out = summarize_episode(pnls, invs)
    map_values = [mean_absolute_position(i) for i in invs if len(i)]
    out["map"] = float(np.mean(map_values)) if map_values else 0.0
    out["total_pnl"] = float(np.sum(pnls)) if pnls else 0.0
    out["max_abs_inventory"] = float(max((np.max(np.abs(i)) for i in invs if len(i)), default=0.0))
    ratio = pnl_over_map(out["mean_pnl"], out["map"])
    if ratio is not None:
        out["pnl_over_map"] = float(ratio)
    if len(pnls) >= 2 and out["std_pnl"] > 0:
        out["sharpe_like"] = float(out["mean_pnl"] / out["std_pnl"])

    steps = backtest_result.get("steps") or []
    all_steps = [s for ep in steps for s in ep]
    fills = [f for s in all_steps for f in s["fills"]]
    out["n_fills"] = float(len(fills))
    filled_vol = float(sum(f["volume"] for f in fills))
    if all_steps and all("fees_paid" in s and "rebates_received" in s for s in all_steps):
        # Prefer the env's own per-fill accounting as logged per step.
        out["fees_paid"] = float(sum(s["fees_paid"] for s in all_steps))
        out["rebates_received"] = float(sum(s["rebates_received"] for s in all_steps))
    else:
        # Fallback for older runs whose step logs lack the fields: recompute from config rates.
        out["fees_paid"] = fee * filled_vol
        out["rebates_received"] = rebate * filled_vol
    spread_capture = float(sum(
        f["volume"] * ((f["mid"] - f["price"]) if f["side"] == "bid" else (f["price"] - f["mid"])) for f in fills))
    out["spread_capture_pnl"] = spread_capture
    out["inventory_pnl"] = out["total_pnl"] - spread_capture + out["fees_paid"] - out["rebates_received"]

    if not all_steps:
        return out

    dds = []
    for ep in steps:
        if ep:
            dds.append(max_drawdown(np.array([0.0] + [s["mtm_pnl"] for s in ep])))
    out["max_drawdown"] = float(max(dds)) if dds else 0.0

    placed = sum(s["placed_volume"] for s in all_steps)
    canceled = sum(s["canceled_volume"] for s in all_steps)
    if placed > 0:
        out["cancel_ratio"] = float(canceled / placed)
        out["fill_rate"] = float(filled_vol / placed)

    spreads, dists = [], []
    for s in all_steps:
        bids = [p for p, v in zip(s["bid_prices"], s["bid_volumes"]) if v > 0]
        asks = [p for p, v in zip(s["ask_prices"], s["ask_volumes"]) if v > 0]
        if bids and asks:
            spreads.append((min(asks) - max(bids)) / tick)
        dists += [abs(p - s["mid"]) / tick for p, v in zip(s["bid_prices"] + s["ask_prices"],
                                                            s["bid_volumes"] + s["ask_volumes"]) if v > 0]
    if spreads:
        out["avg_quoted_spread"] = float(np.mean(spreads))
    if dists:
        out["avg_quote_distance"] = float(np.mean(dists))

    adv = adverse_selection_from_steps(steps, horizon, thr_ticks * tick)
    if adv is not None:
        out["adverse_selection_ratio"] = float(adv)
    return out
