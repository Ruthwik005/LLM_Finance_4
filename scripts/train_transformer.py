import argparse
from market_maker.config import load_config
from market_maker.data.lob_loader import load_lob
from market_maker.data.validators import validate_lob
from market_maker.data.splits import chronological_split
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.temporal_dataset import build_windows
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.transformer_train import train_transformer
from market_maker.models.checkpoints import save_checkpoint
from market_maker.utils.seeds import set_global_seed
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="artifacts/transformer/checkpoint.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)

    df = load_lob(cfg.data.lob_path)
    df, _ = validate_lob(df)
    splits = chronological_split(df, "timestamp", **cfg.data.split.model_dump())
    train_feats = compute_lob_features(splits["train"], depth=cfg.data.depth, return_lags=[1, 5],
                                       vol_window=cfg.baselines.volatility_window)
    val_feats = compute_lob_features(splits["validation"], depth=cfg.data.depth, return_lags=[1, 5],
                                     vol_window=cfg.baselines.volatility_window)

    feature_cols = [c for c in train_feats.columns if c != "timestamp"]
    X_train, y_train = build_windows(train_feats, feature_cols, cfg.transformer.sequence_length,
                                      cfg.transformer.horizons_minutes, tick_threshold=0.0005)
    X_val, y_val = build_windows(val_feats, feature_cols, cfg.transformer.sequence_length,
                                  cfg.transformer.horizons_minutes, tick_threshold=0.0005)

    model = TemporalTransformer(
        input_dim=len(feature_cols), d_model=cfg.transformer.d_model, nhead=cfg.transformer.nhead,
        num_layers=cfg.transformer.num_layers, dim_feedforward=cfg.transformer.dim_feedforward,
        dropout=cfg.transformer.dropout, horizons=cfg.transformer.horizons_minutes,
    )
    history = train_transformer(model, X_train, y_train, X_val, y_val,
                                 epochs=cfg.transformer.epochs, batch_size=cfg.transformer.batch_size, lr=1e-3)
    logger.info("Transformer training finished: %s", history)

    save_checkpoint(model, args.out, extra={
        "feature_cols": feature_cols,
        "sequence_length": cfg.transformer.sequence_length,
        "horizons_minutes": cfg.transformer.horizons_minutes,
        "history": history,
    })
    logger.info("Saved transformer checkpoint to %s", args.out)


if __name__ == "__main__":
    main()
