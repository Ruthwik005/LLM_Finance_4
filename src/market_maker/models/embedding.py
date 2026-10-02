"""Frozen-Transformer inference: embeddings and per-horizon probabilities for every market row."""
from __future__ import annotations
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from market_maker.models.checkpoints import load_checkpoint
from market_maker.models.transformer_encoder import TemporalTransformer


def load_frozen_transformer(path: str, device: str = "cpu"):
    """Rebuild the TemporalTransformer from a checkpoint saved by scripts/train_transformer.py.

    Returns (model, extra) with model in eval mode and requires_grad=False on every parameter."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kw = payload["extra"]["model_kwargs"]
    model, extra = load_checkpoint(TemporalTransformer, path, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), extra


@torch.no_grad()
def embed_all(model, features: np.ndarray, seq_len: int, batch_size: int, device: str = "cpu"):
    """Embedding + horizon probabilities for EVERY row i, from the window of rows [i-seq_len+1, i].

    Rows before the first full window (i < seq_len-1) use a window front-padded by repeating row 0
    (edge padding) and are flagged ``signal_valid = 0``. Output row i is aligned with input row i.
    Batched, eval mode, no grad (the model is frozen). Memory: O(batch * seq_len * F) beyond the outputs.

    Returns (embeddings (n, d_model) float32, {horizon: probs (n, 3) float32}, signal_valid (n,) float32).
    """
    was_training = model.training
    model.eval()
    n = len(features)
    feats = np.asarray(features, dtype=np.float32)
    padded = np.concatenate([np.repeat(feats[:1], seq_len - 1, axis=0), feats], axis=0)
    view = sliding_window_view(padded, seq_len, axis=0)  # (n, F, L), zero-copy
    emb = None
    probs = {h: np.zeros((n, 3), dtype=np.float32) for h in model.horizons}
    for s in range(0, n, batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(view[s:s + batch_size].transpose(0, 2, 1))).to(device)
        pooled, logits = model(batch)
        if emb is None:
            emb = np.zeros((n, pooled.shape[1]), dtype=np.float32)
        emb[s:s + len(batch)] = pooled.cpu().numpy()
        for h in model.horizons:
            probs[h][s:s + len(batch)] = torch.softmax(logits[h], dim=-1).cpu().numpy()
    valid = (np.arange(n) >= seq_len - 1).astype(np.float32)
    if was_training:
        model.train()
    return emb, probs, valid
