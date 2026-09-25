import numpy as np
import pandas as pd
from market_maker.features.fusion import build_fused_dataset


def test_build_fused_dataset_shape_and_clipping():
    n = 5
    lob_feats = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC"),
                               "midprice": [100.0] * n, "spread": [1e6] * n})  # extreme value to test clipping
    aligned_sentiment = pd.DataFrame({"sentiment_ewm_5": [0.5] * n, "article_count_5": [1] * n})
    transformer_embeddings = np.random.randn(n, 4).astype(np.float32)
    transformer_probs = {1: np.random.dirichlet([1, 1, 1], size=n).astype(np.float32)}

    fused, columns = build_fused_dataset(
        lob_feats, aligned_sentiment, transformer_embeddings, transformer_probs,
        private_state_placeholder_dim=3,
    )
    expected_dim = 2 + 2 + 4 + 3 + 3  # midprice+spread, sentiment cols, embedding, probs, private placeholder
    assert fused.shape == (n, expected_dim)
    assert fused.dtype == np.float32
    assert fused.max() <= 10.0 and fused.min() >= -10.0
    assert len(columns) == expected_dim
