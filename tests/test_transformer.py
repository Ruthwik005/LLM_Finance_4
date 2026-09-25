import numpy as np
import torch
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.temporal_dataset import build_windows
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.checkpoints import save_checkpoint, load_checkpoint


def _feature_cols(feats):
    return [c for c in feats.columns if c != "timestamp"]


def test_build_windows_shapes_are_correct():
    df = generate_synthetic_lob(n_rows=2000, depth=5, symbol="BTCUSDT", seed=3)
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    cols = _feature_cols(feats)
    X, y_by_horizon = build_windows(feats, cols, sequence_length=32, horizons_minutes=[1, 5], tick_threshold=0.0005)
    assert X.shape[1] == 32
    assert X.shape[2] == len(cols)
    for h, y in y_by_horizon.items():
        assert y.shape[0] == X.shape[0]
        assert set(np.unique(y)).issubset({0, 1, 2})


def test_transformer_forward_pass_shapes():
    model = TemporalTransformer(input_dim=10, d_model=16, nhead=2, num_layers=1,
                                 dim_feedforward=32, dropout=0.1, horizons=[1, 5])
    x = torch.randn(4, 32, 10)
    pooled, logits_by_horizon = model(x)
    assert pooled.shape == (4, 16)
    assert logits_by_horizon[1].shape == (4, 3)
    assert logits_by_horizon[5].shape == (4, 3)


def test_checkpoint_save_and_load_round_trip(tmp_path):
    model = TemporalTransformer(input_dim=10, d_model=16, nhead=2, num_layers=1,
                                 dim_feedforward=32, dropout=0.1, horizons=[1])
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, path, extra={"feature_cols": ["a", "b"]})
    loaded_model, extra = load_checkpoint(
        TemporalTransformer, path,
        input_dim=10, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.1, horizons=[1],
    )
    assert extra["feature_cols"] == ["a", "b"]
    x = torch.randn(2, 32, 10)
    pooled, _ = loaded_model(x)
    assert pooled.shape == (2, 16)
