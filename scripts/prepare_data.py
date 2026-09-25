import argparse
import os
from market_maker.config import load_config
from market_maker.data.lob_loader import load_lob
from market_maker.data.news_loader import load_news
from market_maker.data.validators import validate_lob, validate_news
from market_maker.data.splits import chronological_split
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if not os.path.exists(cfg.data.lob_path):
        raise FileNotFoundError(
            f"LOB data not found at {cfg.data.lob_path}. Run `make sample-data` first, "
            f"or place real Mode-B data (crypto LOB) at this path."
        )

    lob_df = load_lob(cfg.data.lob_path)
    lob_df, lob_report = validate_lob(lob_df)
    splits = chronological_split(lob_df, "timestamp", **cfg.data.split.model_dump())
    logger.info("Prepared LOB: %d train / %d val / %d test rows. Validation report: %s",
                len(splits["train"]), len(splits["validation"]), len(splits["test"]), lob_report)

    if os.path.exists(cfg.data.news_path):
        news_df = load_news(cfg.data.news_path)
        news_df, news_report = validate_news(news_df)
        logger.info("Prepared news: %d rows. Validation report: %s", len(news_df), news_report)
    else:
        logger.info("No news file at %s — proceeding in news-optional mode.", cfg.data.news_path)


if __name__ == "__main__":
    main()
