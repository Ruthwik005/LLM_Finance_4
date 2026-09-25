from __future__ import annotations
import numpy as np
import pandas as pd


def _label_from_return(ret: float, threshold: float) -> int:
    if ret > threshold:
        return 2  # up
    if ret < -threshold:
        return 0  # down
    return 1  # flat


def build_windows(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    horizons_minutes: list[int],
    tick_threshold: float,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    values = feature_df[feature_cols].to_numpy(dtype=np.float32)
    timestamps = pd.to_datetime(feature_df["timestamp"], utc=True).to_numpy()
    midprice = feature_df["midprice"].to_numpy() if "midprice" in feature_df.columns else None

    n = len(feature_df)
    max_horizon_ns = max(horizons_minutes) * 60 * 1e9

    windows = []
    labels_by_horizon: dict[int, list[int]] = {h: [] for h in horizons_minutes}

    for end in range(sequence_length, n):
        window_start = end - sequence_length
        t_end = timestamps[end - 1]
        if (t_end - timestamps[-1]).astype("int64") if False else False:
            pass
        # skip windows too close to the end of the series to have future labels
        time_to_end_ns = (timestamps[-1] - t_end) / np.timedelta64(1, 'ns')
        if time_to_end_ns < max_horizon_ns:
            continue

        window = values[window_start:end]
        valid = True
        horizon_labels = {}
        for h in horizons_minutes:
            target_time = t_end + np.timedelta64(h, "m")
            future_idx = np.searchsorted(timestamps, target_time)
            if future_idx >= n or midprice is None:
                valid = False
                break
            ret = (midprice[future_idx] - midprice[end - 1]) / midprice[end - 1]
            horizon_labels[h] = _label_from_return(ret, tick_threshold)
        if not valid:
            continue

        windows.append(window)
        for h in horizons_minutes:
            labels_by_horizon[h].append(horizon_labels[h])

    X = np.stack(windows, axis=0) if windows else np.zeros((0, sequence_length, len(feature_cols)), dtype=np.float32)
    y_by_horizon = {h: np.array(v, dtype=np.int64) for h, v in labels_by_horizon.items()}
    return X, y_by_horizon
