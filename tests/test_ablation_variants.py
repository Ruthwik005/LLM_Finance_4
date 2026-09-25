import numpy as np
import pandas as pd
from market_maker.features.fusion import build_fused_dataset


def test_lob_only_variant_excludes_sentiment_and_transformer_columns():
    """
    Verify that LOB_ONLY variant produces fused dataset with no sentiment/transformer columns.
    This tests that build_fused_dataset correctly excludes columns when given empty inputs.
    """
    n = 4
    lob_feats = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC"),
        "midprice": [1.0] * n
    })
    empty_sentiment = pd.DataFrame(index=range(n))  # LOB_ONLY variant passes empty sentiment frame
    empty_embeddings = np.zeros((n, 0), dtype=np.float32)
    fused, columns = build_fused_dataset(
        lob_feats, empty_sentiment, empty_embeddings, {},
        private_state_placeholder_dim=2
    )
    assert not any("sentiment" in c for c in columns)
    assert not any("transformer" in c for c in columns)
    assert fused.shape == (n, 1 + 0 + 0 + 2)
