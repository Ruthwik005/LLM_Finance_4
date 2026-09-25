"""Tests for configuration loader."""
from market_maker.config import load_config


def test_load_default_config_has_required_sections():
    cfg = load_config("configs/default.yaml")
    assert cfg.project.seed == 42
    assert cfg.data.symbol == "BTCUSDT"
    assert cfg.simulator.levels_per_side == 2
    assert cfg.sac.total_timesteps > 0


def test_fast_demo_uses_smaller_budget_than_default():
    default = load_config("configs/default.yaml")
    fast = load_config("configs/fast_demo.yaml")
    assert fast.sac.total_timesteps < default.sac.total_timesteps
    assert fast.transformer.epochs < default.transformer.epochs
