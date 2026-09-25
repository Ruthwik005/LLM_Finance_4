from __future__ import annotations

ABLATION_VARIANTS = ["SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL"]

_VARIANT_TO_FLAG = {
    "SAC_LOB_ONLY": "run_lob_only",
    "SAC_LOB_TRANSFORMER": "run_lob_transformer",
    "SAC_LOB_SENTIMENT": "run_lob_sentiment",
    "SAC_FULL": "run_full",
}


def select_enabled_variants(experiments_config: dict) -> list[str]:
    return [v for v in ABLATION_VARIANTS if experiments_config.get(_VARIANT_TO_FLAG[v], False)]
