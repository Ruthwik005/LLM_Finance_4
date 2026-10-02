from __future__ import annotations
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

from market_maker.utils.io import file_sha256

PACKAGES = ["torch", "numpy", "pandas", "stable-baselines3", "gymnasium"]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def build_run_metadata(cfg, *, data_mode: str, model_path: str, variant: str, splits: dict | None = None,
                       ts_col: str = "timestamp") -> dict:
    """Run metadata dict. ``cfg`` is an AppConfig; ``splits`` maps split name -> DataFrame (for date ranges)."""
    ranges = {}
    for name, df in (splits or {}).items():
        if len(df):
            ranges[name] = {"start": str(df[ts_col].min()), "end": str(df[ts_col].max()), "rows": int(len(df))}
    return {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.project.seed,
        "python_version": sys.version.split()[0],
        "package_versions": {p: _pkg_version(p) for p in PACKAGES},
        "dataset_fingerprints": {"lob_sha256": file_sha256(cfg.data.lob_path),
                                 "news_sha256": file_sha256(cfg.data.news_path)},
        "date_ranges": ranges,
        "simulator_mode": cfg.simulator.execution_mode,
        "reference_price_mode": cfg.simulator.reference_price_mode,
        "reward_config": cfg.reward.model_dump(),
        "feature_config": {"depth": cfg.data.depth, "return_lags": cfg.features.return_lags,
                           "volatility_window": cfg.features.vol_window,
                           "clip_value": cfg.features.clip_value,
                           "label_threshold_bps": cfg.features.label_threshold_bps,
                           "sentiment": cfg.sentiment.model_dump(), "transformer": cfg.transformer.model_dump()},
        "data_mode": data_mode,
        "model_path": model_path,
        "variant": variant,
    }
