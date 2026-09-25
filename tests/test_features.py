import numpy as np
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.features.lob_features import compute_lob_features, fit_scaler, apply_scaler


def test_compute_lob_features_has_no_nan_or_inf():
    df = generate_synthetic_lob(n_rows=500, depth=5, symbol="BTCUSDT", seed=7)
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    numeric = feats.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


def test_midprice_and_spread_are_correct():
    df = generate_synthetic_lob(n_rows=10, depth=1, symbol="BTCUSDT", seed=7)
    feats = compute_lob_features(df, depth=1, return_lags=[1])
    expected_mid = (df["bid_price_1"] + df["ask_price_1"]) / 2
    expected_spread = df["ask_price_1"] - df["bid_price_1"]
    assert np.allclose(feats["midprice"], expected_mid)
    assert np.allclose(feats["spread"], expected_spread)


def test_scaler_fit_on_train_reused_on_validation():
    df = generate_synthetic_lob(n_rows=1000, depth=5, symbol="BTCUSDT", seed=7)
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    feature_cols = ["midprice", "spread"]
    train_feats, val_feats = feats.iloc[:700], feats.iloc[700:]
    scaler = fit_scaler(train_feats, feature_cols)
    scaled_val = apply_scaler(val_feats, feature_cols, scaler)
    assert abs(scaled_val["midprice"].mean()) < 5  # reused scaler, not refit on val


def test_compute_lob_features_preserves_timezone_awareness():
    """Verify that compute_lob_features preserves tz-aware timestamp column."""
    df = generate_synthetic_lob(n_rows=500, depth=5, symbol="BTCUSDT", seed=42)
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    # The returned timestamp column must be tz-aware (UTC)
    assert feats["timestamp"].dt.tz is not None
    assert str(feats["timestamp"].dt.tz) == "UTC"
