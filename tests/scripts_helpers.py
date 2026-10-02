"""Shared test helper: tiny Transformer training through the same code path scripts/train_transformer.py uses."""
import numpy as np
import pandas as pd

from market_maker.features.pipeline import base_features_path, load_meta, lob_split_path, transformer_checkpoint_path, \
    transformer_binding
from market_maker.features.temporal_dataset import WindowDataset
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.transformer_train import fit_transformer
from market_maker.utils.seeds import set_global_seed


def train_tiny_transformer(cfg, art):
    t = cfg.transformer
    meta = load_meta(art)

    def ds(split):
        X = np.load(base_features_path(art, split))
        lob = pd.read_parquet(lob_split_path(art, split), columns=["timestamp", "bid_price_1", "ask_price_1"])
        mid = ((lob["bid_price_1"] + lob["ask_price_1"]) / 2).to_numpy()
        return WindowDataset(X, lob["timestamp"], mid, t.sequence_length, t.horizons_minutes,
                             cfg.features.label_threshold_bps, stride=t.window_stride,
                         max_gap_factor=t.label_max_gap_factor)

    set_global_seed(0)
    kw = dict(input_dim=len(meta["lob_feature_columns"]), d_model=t.d_model, nhead=t.nhead, num_layers=t.num_layers,
              dim_feedforward=t.dim_feedforward, dropout=t.dropout, horizons=t.horizons_minutes)
    model = TemporalTransformer(**kw)
    extra = {"feature_cols": meta["lob_feature_columns"], "sequence_length": t.sequence_length,
             "horizons_minutes": t.horizons_minutes, "model_kwargs": kw, **transformer_binding(meta)}
    return fit_transformer(model, ds("train"), ds("validation"), epochs=2, batch_size=32, lr=1e-3, weight_decay=0.01,
                           patience=2, device="cpu", checkpoint_path=transformer_checkpoint_path(art),
                           checkpoint_extra=extra, use_class_weights=True)
