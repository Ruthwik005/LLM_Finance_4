import shutil
import numpy as np
import pandas as pd
import pytest

from market_maker.config import load_config
from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news


def make_wp2_cfg(tmp_path, n_rows=6000):
    """Tiny config over synthetic data spanning ~9 days (2-minute spacing) so the day-based split has all 3 parts."""
    lob = generate_synthetic_lob(n_rows=n_rows, depth=5, symbol="BTCUSDT", seed=7)
    lob["timestamp"] = pd.date_range("2024-01-01", periods=n_rows, freq="2min", tz="UTC")
    news = generate_synthetic_news(n_items=120, symbol="BTCUSDT", lob_df=lob, seed=7)
    lp, npth = str(tmp_path / "lob.parquet"), str(tmp_path / "news.parquet")
    lob.to_parquet(lp)
    news.to_parquet(npth)
    cfg = load_config("configs/fast_demo.yaml")
    cfg.data.lob_path, cfg.data.news_path = lp, npth
    cfg.simulator.episode_minutes = 1  # 120 steps
    cfg.transformer.sequence_length = 16
    cfg.transformer.d_model = 16
    cfg.transformer.epochs = 2
    cfg.transformer.window_stride = 2
    cfg.sac.total_timesteps, cfg.sac.learning_starts, cfg.sac.eval_freq = 300, 50, 150
    cfg.sac.buffer_size, cfg.sac.batch_size, cfg.sac.net_arch, cfg.sac.n_eval_episodes = 1000, 32, [32, 32], 2
    cfg.experiment.n_test_episodes = 3
    return cfg


@pytest.fixture(scope="session")
def wp2_built(tmp_path_factory):
    """Session-scoped: synthetic data -> base features (stub sentiment) -> tiny Transformer -> all 4 variants."""
    from market_maker.features.pipeline import (build_base_features, build_variant_features,
                                                 load_meta, transformer_checkpoint_path)
    from scripts_helpers import train_tiny_transformer
    root = tmp_path_factory.mktemp("wp2")
    cfg = make_wp2_cfg(root)
    art = str(root / "artifacts")
    build_base_features(cfg, art, stub_sentiment=True)
    train_tiny_transformer(cfg, art)
    res = build_variant_features(cfg, art, device="cpu")
    assert not res["skipped"], res
    return cfg, art


@pytest.fixture()
def wp2_copy(wp2_built, tmp_path):
    """Per-test private copy of the built artifacts."""
    cfg, art = wp2_built
    dst = str(tmp_path / "art")
    shutil.copytree(art, dst)
    return cfg, dst
