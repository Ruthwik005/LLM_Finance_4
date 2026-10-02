from __future__ import annotations
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


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


# ---------------------------------------------------------------------------
# Memory-safe, vectorized path (production). `build_windows` above is the slow reference.
# ---------------------------------------------------------------------------
IGNORE_LABEL = -100  # label of a (window, horizon) whose future horizon extends past the data


def _to_ns_array(timestamps) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")


def compute_horizon_labels(timestamps, midprice: np.ndarray, horizons_minutes: list[int],
                           threshold_bps: float, max_gap_factor: float = 1.5) -> dict[int, np.ndarray]:
    """Per-ROW label for each horizon h: class of the return from row i to the first row with
    timestamp >= t_i + h minutes (0 = down, 1 = flat, 2 = up; strict thresholds like the reference).
    Rows whose horizon target lies beyond the last timestamp, or whose target row is more than
    `h * max_gap_factor` after t_i (the horizon spans a session gap / overnight break), get IGNORE_LABEL.
    Vectorized via searchsorted.
    """
    ts = _to_ns_array(timestamps)
    mid = np.asarray(midprice, dtype=np.float64)
    n = len(ts)
    thr = threshold_bps / 1e4
    out = {}
    for h in horizons_minutes:
        fi = np.searchsorted(ts, ts + int(h) * 60_000_000_000, side="left")
        valid = fi < n
        h_ns = int(h) * 60_000_000_000
        elapsed = ts[np.minimum(fi, n - 1)] - ts
        valid &= elapsed <= h_ns * float(max_gap_factor)
        fut = mid[np.minimum(fi, n - 1)]
        ret = (fut - mid) / mid
        lab = np.where(ret > thr, 2, np.where(ret < -thr, 0, 1)).astype(np.int64)
        lab[~valid] = IGNORE_LABEL
        out[h] = lab
    return out


class WindowDataset(Dataset):
    """Lazily sliced sliding windows over a contiguous (n, F) feature array.

    Sample k covers rows [end-L, end) with end = L + k*stride; its label for horizon h is the per-row label of the
    window's last row (end-1). Nothing of shape (n, L, F) is ever materialised. Windows where every horizon is
    unlabeled (too close to the end of the data) are dropped; partially labeled windows keep IGNORE_LABEL entries.
    """

    def __init__(self, features: np.ndarray, timestamps, midprice: np.ndarray, sequence_length: int,
                 horizons_minutes: list[int], threshold_bps: float, stride: int = 1,
                 max_gap_factor: float = 1.5):
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.seq_len = int(sequence_length)
        self.horizons = list(horizons_minutes)
        n = len(self.features)
        row_labels = compute_horizon_labels(timestamps, midprice, self.horizons, threshold_bps, max_gap_factor)
        ends = np.arange(self.seq_len, n + 1, max(1, int(stride)))
        lab = np.stack([row_labels[h][ends - 1] for h in self.horizons], axis=1) if len(ends) else \
            np.zeros((0, len(self.horizons)), dtype=np.int64)
        keep = (lab != IGNORE_LABEL).any(axis=1)
        self.ends = ends[keep]
        self.labels = lab[keep]  # (n_samples, n_horizons)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int):
        end = int(self.ends[i])
        return torch.from_numpy(self.features[end - self.seq_len:end].copy()), torch.from_numpy(self.labels[i])

    def label_counts(self) -> dict[int, np.ndarray]:
        """Class counts (3,) per horizon over valid labels (for class weights / majority baselines)."""
        return {h: np.bincount(self.labels[:, j][self.labels[:, j] >= 0], minlength=3)
                for j, h in enumerate(self.horizons)}
