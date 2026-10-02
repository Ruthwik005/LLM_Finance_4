"""Train the temporal Transformer on the TRAIN split (validation split for early stopping; test never touched).

Requires `scripts/build_features.py --stage base` first. Everything is config driven (transformer.* and
features.label_threshold_bps); CLI flags only override.
"""
import argparse
import os
import numpy as np
import pandas as pd
from market_maker.config import load_config
from market_maker.features.pipeline import (SPLITS, base_features_path, load_meta, lob_split_path,
                                             transformer_checkpoint_path, features_dir)
from market_maker.features.temporal_dataset import WindowDataset
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.transformer_train import fit_transformer, resolve_device
from market_maker.utils.seeds import set_global_seed
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def _dataset(cfg, artifacts_dir, split):
    X = np.load(base_features_path(artifacts_dir, split))
    lob = pd.read_parquet(lob_split_path(artifacts_dir, split), columns=["timestamp", "bid_price_1", "ask_price_1"])
    mid = ((lob["bid_price_1"] + lob["ask_price_1"]) / 2).to_numpy()
    t = cfg.transformer
    return WindowDataset(X, lob["timestamp"], mid, t.sequence_length, t.horizons_minutes,
                         cfg.features.label_threshold_bps, stride=t.window_stride)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--out", default=None, help="checkpoint path (default <artifacts>/transformer/checkpoint.pt)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    t = cfg.transformer
    set_global_seed(cfg.project.seed)
    meta = load_meta(args.artifacts_dir)
    cols = meta["lob_feature_columns"]
    train_ds, val_ds = _dataset(cfg, args.artifacts_dir, "train"), _dataset(cfg, args.artifacts_dir, "validation")
    logger.info("windows: train=%d val=%d (stride %d)  label counts train=%s", len(train_ds), len(val_ds),
                t.window_stride, train_ds.label_counts())

    model_kwargs = dict(input_dim=len(cols), d_model=t.d_model, nhead=t.nhead, num_layers=t.num_layers,
                        dim_feedforward=t.dim_feedforward, dropout=t.dropout, horizons=t.horizons_minutes)
    model = TemporalTransformer(**model_kwargs)
    out = args.out or transformer_checkpoint_path(args.artifacts_dir)
    extra = {"feature_cols": cols, "sequence_length": t.sequence_length, "horizons_minutes": t.horizons_minutes,
             "scaler_path": meta["scaler_path"], "label_threshold_bps": cfg.features.label_threshold_bps,
             "model_kwargs": model_kwargs, "config": cfg.model_dump(), "seed": cfg.project.seed}
    history = fit_transformer(
        model, train_ds, val_ds, epochs=args.epochs or t.epochs, batch_size=t.batch_size, lr=t.lr,
        weight_decay=t.weight_decay, patience=t.patience, device=resolve_device(args.device or cfg.project.device),
        num_workers=t.num_workers, use_class_weights=t.class_weights, seed=cfg.project.seed,
        checkpoint_path=out, checkpoint_extra=extra,
        curve_path=os.path.join(args.artifacts_dir, "transformer", "training_curve.csv"))
    logger.info("best epoch %d val_loss %.4f val_acc %s (majority-class acc %s)", history["best_epoch"],
                history["best_val_loss"], history["val_acc"][history["best_epoch"]], history["val_majority_acc"])
    logger.info("Saved transformer checkpoint to %s. Next: build_features.py --stage fusion", out)


if __name__ == "__main__":
    main()
