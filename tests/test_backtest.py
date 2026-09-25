import json
import os
import numpy as np
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.ablation import select_enabled_variants, ABLATION_VARIANTS


def _make_env():
    lob_df = generate_synthetic_lob(n_rows=500, depth=2, symbol="BTCUSDT", seed=5)
    fused = np.random.randn(len(lob_df), 8).astype(np.float32)
    return MarketMakingEnv(
        fused_features=fused, raw_lob_df=lob_df, feature_columns=[f"f{i}" for i in range(8)],
        tick_size=0.01, levels_per_side=2, min_spread_ticks=1, max_spread_ticks=8, max_mid_offset_ticks=4,
        total_volume_per_side=20.0, max_inventory=100.0, fee_per_unit=0.0,
        reward_config={"type": "truncated_inventory", "inventory_threshold": 40, "inventory_penalty": 0.001,
                       "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=20,
    )


def test_run_backtest_produces_expected_keys_and_lengths():
    env = _make_env()
    agent = HeuristicMarketMaker()
    result = run_backtest(env, agent, n_episodes=3, is_heuristic=True, max_inventory=100.0)
    assert len(result["episode_pnls"]) == 3
    assert len(result["inventories"]) == 3


def test_write_run_artifacts_creates_required_files(tmp_path):
    env = _make_env()
    agent = HeuristicMarketMaker()
    result = run_backtest(env, agent, n_episodes=2, is_heuristic=True, max_inventory=100.0)
    run_dir = str(tmp_path / "run_001")
    write_run_artifacts(run_dir, config={"seed": 42}, metadata={"seed": 42}, backtest_result=result,
                         metrics={"mean_pnl": 0.0})
    for fname in ["config.yaml", "metadata.json", "metrics.json", "episodes.csv", "fills.csv"]:
        assert os.path.exists(os.path.join(run_dir, fname))
    with open(os.path.join(run_dir, "metadata.json")) as f:
        assert json.load(f)["seed"] == 42


def test_select_enabled_variants_respects_config():
    cfg = {"run_lob_only": True, "run_lob_transformer": False, "run_lob_sentiment": True, "run_full": True,
           "run_heuristic": True, "run_td3": False}
    variants = select_enabled_variants(cfg)
    assert "SAC_LOB_ONLY" in variants
    assert "SAC_LOB_TRANSFORMER" not in variants
    assert "SAC_FULL" in variants
    assert set(ABLATION_VARIANTS) == {"SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL"}
