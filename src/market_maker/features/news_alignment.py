from __future__ import annotations
import numpy as np
import pandas as pd


def align_sentiment_to_market(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lookback_minutes: list[int],
    max_age_minutes: int,
) -> pd.DataFrame:
    result = market_df.copy().reset_index(drop=True)
    news_sorted = news_df.sort_values("published_at").reset_index(drop=True)
    news_times = news_sorted["published_at"].to_numpy()
    news_scores = news_sorted["sentiment_score"].to_numpy() if len(news_sorted) else np.array([])

    for w in lookback_minutes:
        result[f"sentiment_ewm_{w}"] = 0.0
        result[f"sentiment_max_abs_{w}"] = 0.0
        result[f"article_count_{w}"] = 0
    result["minutes_since_last_article"] = float(max_age_minutes)
    result["has_recent_news"] = False

    for idx, t in enumerate(result["timestamp"]):
        # as-of backward join: only news with published_at <= t is visible
        visible_mask = news_times <= t
        n_visible = int(visible_mask.sum())
        if n_visible == 0:
            continue

        last_pub = news_times[visible_mask][-1]
        age_minutes = (t - last_pub) / np.timedelta64(1, "m")
        result.loc[idx, "minutes_since_last_article"] = min(float(age_minutes), float(max_age_minutes))
        result.loc[idx, "has_recent_news"] = bool(age_minutes <= max_age_minutes)

        for w in lookback_minutes:
            window_mask = visible_mask & (news_times > t - np.timedelta64(w, "m"))
            count = int(window_mask.sum())
            result.loc[idx, f"article_count_{w}"] = count
            if count > 0:
                window_scores = news_scores[window_mask]
                weights = np.exp(np.linspace(-1.0, 0.0, count))
                weights /= weights.sum()
                result.loc[idx, f"sentiment_ewm_{w}"] = float(np.dot(weights, window_scores))
                result.loc[idx, f"sentiment_max_abs_{w}"] = float(np.max(np.abs(window_scores)))

    return result
