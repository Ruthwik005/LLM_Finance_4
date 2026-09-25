from __future__ import annotations
import os
from typing import Protocol
import pandas as pd
import torch


class SentimentScorer(Protocol):
    def score(self, texts: list[str]) -> pd.DataFrame:
        ...


class FinbertScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32, device: str = "cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        # ProsusAI/finbert label order: positive, negative, neutral
        self.label_order = ["positive", "negative", "neutral"]

    def score(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    p = dict(zip(self.label_order, row))
                    rows.append(p)
        df = pd.DataFrame(rows)[["positive", "negative", "neutral"]]
        df.columns = ["p_positive", "p_negative", "p_neutral"]
        df["sentiment_score"] = df["p_positive"] - df["p_negative"]
        return df


def score_news_dataframe(news_df: pd.DataFrame, scorer: SentimentScorer) -> pd.DataFrame:
    texts = (news_df["headline"].fillna("") + " " + news_df.get("body", "").fillna("")).tolist() \
        if "body" in news_df.columns else news_df["headline"].fillna("").tolist()
    sentiment = scorer.score(texts).reset_index(drop=True)
    result = news_df.reset_index(drop=True).copy()
    for col in ["p_positive", "p_negative", "p_neutral", "sentiment_score"]:
        result[col] = sentiment[col]
    return result


def cache_sentiment(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)


def load_cached_sentiment(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)
