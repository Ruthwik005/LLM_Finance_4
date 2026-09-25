import argparse
from market_maker.config import load_config
from market_maker.data.news_loader import load_news
from market_maker.data.validators import validate_news
from market_maker.features.sentiment import (
    FinbertScorer, score_news_dataframe, cache_sentiment, load_cached_sentiment,
)
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="artifacts/sentiment/news_sentiment.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cached = load_cached_sentiment(args.out)
    if cached is not None:
        logger.info("Sentiment cache found at %s (%d rows); skipping FinBERT run.", args.out, len(cached))
        return

    news_df = load_news(cfg.data.news_path)
    news_df, report = validate_news(news_df)
    logger.info("Validated news: %d rows, removed=%s", len(news_df), report)

    scorer = FinbertScorer(model_name=cfg.sentiment.model_name, batch_size=cfg.sentiment.batch_size)
    scored = score_news_dataframe(news_df, scorer)
    cache_sentiment(scored, args.out)
    logger.info("Cached %d scored news rows to %s", len(scored), args.out)


if __name__ == "__main__":
    main()
