"""Build all feature artifacts (scaled LOB features, sentiment, Transformer embeddings, per-variant arrays).

Real run order:  build_features.py (base) -> train_transformer.py -> build_features.py --stage fusion
(`--stage all`, the default, does base then fusion; Transformer variants are skipped with a warning
until artifacts/transformer/checkpoint.pt exists).
"""
import argparse
import time
from market_maker.config import load_config
from market_maker.features.pipeline import build_base_features, build_variant_features, SAC_VARIANTS
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--stage", choices=["base", "fusion", "all"], default="all")
    p.add_argument("--variants", nargs="*", default=None, help=f"default: config experiment.variants")
    p.add_argument("--stub-sentiment", action="store_true",
                   help="SMOKE TESTS ONLY: deterministic synthetic scores labeled 'synthetic_stub' instead of FinBERT")
    p.add_argument("--no-sentiment", action="store_true", help="skip sentiment features entirely")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps for FinBERT / Transformer inference")
    args = p.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    if args.stage in ("base", "all"):
        build_base_features(cfg, args.artifacts_dir, with_sentiment=not args.no_sentiment,
                            stub_sentiment=args.stub_sentiment, device=args.device)
        logger.info("base stage done in %.1fs", time.time() - t0)
    if args.stage in ("fusion", "all"):
        variants = args.variants or cfg.experiment.variants or SAC_VARIANTS
        res = build_variant_features(cfg, args.artifacts_dir, variants, device=args.device)
        for v, why in res["skipped"].items():
            logger.warning("SKIPPED %s: %s", v, why)
    logger.info("build_features finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
