import os
import numpy as np
from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news
from market_maker.data.validators import validate_lob, validate_news
from market_maker.data.splits import chronological_split
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.sentiment import score_news_dataframe
from market_maker.features.news_alignment import align_sentiment_to_market
from market_maker.features.temporal_dataset import build_windows
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.transformer_train import train_transformer
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.report import generate_report


class _StubScorer:
    def score(self, texts):
        import pandas as pd
        n = len(texts)
        return pd.DataFrame({"p_positive": [0.5] * n, "p_negative": [0.2] * n,
                              "p_neutral": [0.3] * n, "sentiment_score": [0.3] * n})


def test_full_pipeline_runs_end_to_end_on_synthetic_data(tmp_path):
    lob_df = generate_synthetic_lob(n_rows=3000, depth=5, symbol="BTCUSDT", seed=99)
    news_df = generate_synthetic_news(n_items=30, symbol="BTCUSDT", lob_df=lob_df, seed=99)

    lob_df, _ = validate_lob(lob_df)
    news_df, _ = validate_news(news_df)
    splits = chronological_split(lob_df, "timestamp", 0.70, 0.15, 0.15)

    train_feats = compute_lob_features(splits["train"], depth=5, return_lags=[1, 5])
    # Fix timezone mismatch: compute_lob_features loses timezone info, so restore it from the source data
    import pandas as pd
    train_feats["timestamp"] = pd.to_datetime(train_feats["timestamp"], utc=True)
    scored_news = score_news_dataframe(news_df, _StubScorer())
    aligned = align_sentiment_to_market(train_feats, scored_news, lookback_minutes=[5, 30], max_age_minutes=240)
    assert not aligned.isna().any().any()

    feature_cols = [c for c in train_feats.columns if c != "timestamp"]
    X, y_by_horizon = build_windows(train_feats, feature_cols, sequence_length=16,
                                     horizons_minutes=[1], tick_threshold=0.0005)
    assert X.shape[0] > 0
    model = TemporalTransformer(input_dim=len(feature_cols), d_model=16, nhead=2, num_layers=1,
                                 dim_feedforward=32, dropout=0.1, horizons=[1])
    split_point = int(len(X) * 0.8)
    history = train_transformer(model, X[:split_point], {1: y_by_horizon[1][:split_point]},
                                 X[split_point:], {1: y_by_horizon[1][split_point:]},
                                 epochs=1, batch_size=16, lr=1e-3)
    assert len(history["train_loss"]) == 1

    fused = train_feats[feature_cols].to_numpy(dtype=np.float32)
    env = MarketMakingEnv(
        fused_features=fused, raw_lob_df=splits["train"].reset_index(drop=True), feature_columns=feature_cols,
        tick_size=0.01, levels_per_side=2, min_spread_ticks=1, max_spread_ticks=8, max_mid_offset_ticks=4,
        total_volume_per_side=20.0, max_inventory=100.0, fee_per_unit=0.0,
        reward_config={"type": "truncated_inventory", "inventory_threshold": 40, "inventory_penalty": 0.001,
                       "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=20,
    )
    agent = HeuristicMarketMaker()
    result = run_backtest(env, agent, n_episodes=2, is_heuristic=True, max_inventory=100.0)
    assert len(result["episode_pnls"]) == 2

    run_dir = str(tmp_path / "artifacts" / "evaluation" / "smoke_run")
    write_run_artifacts(run_dir, config={"seed": 99}, metadata={"seed": 99, "model_path": "N/A"},
                         backtest_result=result, metrics={"mean_pnl": float(np.mean(result["episode_pnls"]))})

    report_path = str(tmp_path / "artifacts" / "reports" / "FINAL_PROJECT_REPORT.md")
    text = generate_report(runs_root=str(tmp_path / "artifacts" / "evaluation"), output_path=report_path)
    assert os.path.exists(report_path)
    assert "smoke_run" in text
