from __future__ import annotations
import numpy as np
import pandas as pd


def build_fused_dataset(
    lob_feats: pd.DataFrame,
    aligned_sentiment: pd.DataFrame,
    transformer_embeddings: np.ndarray,
    transformer_probs: dict[int, np.ndarray],
    private_state_placeholder_dim: int = 0,
    signal_valid: np.ndarray | None = None,
    clip_value: float = 10.0,
) -> tuple[np.ndarray, list[str]]:
    lob_cols = [c for c in lob_feats.columns if c != "timestamp"]
    lob_array = lob_feats[lob_cols].to_numpy(dtype=np.float32)

    sentiment_cols = list(aligned_sentiment.columns)
    sentiment_array = aligned_sentiment[sentiment_cols].to_numpy(dtype=np.float32)

    prob_cols = []
    prob_arrays = []
    for horizon in sorted(transformer_probs.keys()):
        probs = transformer_probs[horizon].astype(np.float32)
        prob_arrays.append(probs)
        prob_cols.extend([f"transformer_prob_h{horizon}_{cls}" for cls in ("down", "flat", "up")])
    prob_array = np.concatenate(prob_arrays, axis=1) if prob_arrays else np.zeros((len(lob_feats), 0), dtype=np.float32)

    embedding_cols = [f"transformer_embed_{i}" for i in range(transformer_embeddings.shape[1])]
    private_cols = [f"private_placeholder_{i}" for i in range(private_state_placeholder_dim)]
    private_array = np.zeros((len(lob_feats), private_state_placeholder_dim), dtype=np.float32)

    valid_cols: list[str] = []
    valid_array = np.zeros((len(lob_feats), 0), dtype=np.float32)
    if signal_valid is not None:
        valid_cols = ["transformer_signal_valid"]
        valid_array = np.asarray(signal_valid, dtype=np.float32).reshape(-1, 1)

    # Note: the env appends its own private state, so callers pass private_state_placeholder_dim=0.
    fused = np.concatenate(
        [lob_array, sentiment_array, transformer_embeddings.astype(np.float32), prob_array, valid_array,
         private_array],
        axis=1,
    )
    fused = np.clip(fused, -clip_value, clip_value).astype(np.float32)

    columns = lob_cols + sentiment_cols + embedding_cols + prob_cols + valid_cols + private_cols
    return fused, columns
