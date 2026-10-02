"""Pipeline orchestration for scripts/run_all.py.

Stages (in order): prepare_data -> build_features -> train_transformer -> run_experiments -> generate_report.
  * build_features   = build_features.py --stage base (real FinBERT unless stub_sentiment is explicitly requested)
  * train_transformer = train_transformer.py, then build_features.py --stage fusion (embeddings + per-variant arrays)
  * run_experiments  = run_experiments.py (variants x seeds + HEURISTIC / AVELLANEDA_STOIKOV baselines)
  * generate_report  = generate_report.py
News is optional: if no news file (and no cached FinBERT scores) exists, sentiment features are skipped and the sentiment
variants are recorded as N/A with the reason in <artifacts>/logs/pipeline_status.json (the report reads it).
Every stage is a subprocess run with sys.executable; the first failure stops the pipeline with a non-zero exit code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from market_maker.features.pipeline import VARIANT_USES
from market_maker.training.experiment import default_experiment_id
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = ["prepare_data", "build_features", "train_transformer", "run_experiments", "generate_report"]
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


class PipelineError(RuntimeError):
    pass


def _script(name: str) -> str:
    return str(SCRIPTS_DIR / name)


def sentiment_availability(cfg, artifacts_root: str, stub: bool) -> tuple[bool, str | None]:
    """(available, reason_if_not). Real FinBERT needs the news file or cached FinBERT scores; the stub needs the news file."""
    if not cfg.sentiment.enabled:
        return False, "sentiment is disabled in the config (sentiment.enabled: false)"
    has_news = os.path.exists(cfg.data.news_path)
    cache = os.path.join(artifacts_root, "sentiment", "news_sentiment.parquet")
    if has_news or (not stub and os.path.exists(cache)):
        return True, None
    return False, f"no news file at {cfg.data.news_path} (news-optional mode: sentiment variants not run)"


def build_plan(cfg, args) -> dict:
    """Pure function: turns parsed run_all args + config into an ordered plan (no side effects)."""
    root = args.artifacts_root
    stub = bool(args.stub_sentiment)
    variants = [v.upper() for v in (args.variants or cfg.experiment.variants)]
    skipped: dict[str, str] = {}
    sent_ok, sent_why = sentiment_availability(cfg, root, stub)
    for v in list(variants):
        uses_s, uses_t = VARIANT_USES[v]
        if uses_s and not sent_ok:
            skipped[v] = sent_why
        elif uses_t and not cfg.transformer.enabled:
            skipped[v] = "transformer is disabled in the config (transformer.enabled: false)"
    variants = [v for v in variants if v not in skipped]
    if not variants:
        raise PipelineError(f"no runnable variants left; skipped: {skipped}")
    py, cfgp = sys.executable, args.config
    base = [py, _script("build_features.py"), "--config", cfgp, "--artifacts-dir", root]
    base_flags = (["--stub-sentiment"] if stub else []) if sent_ok else ["--no-sentiment"]
    var_flags = ["--variants", *variants]
    exp = [py, _script("run_experiments.py"), "--config", cfgp, "--artifacts-root", root]
    exp += ["--variants", *variants]
    if args.seeds:
        exp += ["--seeds", *[str(s) for s in args.seeds]]
    exp_id = args.experiment_id or default_experiment_id(cfg, args.fast)
    exp += ["--experiment-id", exp_id]
    for flag, on in (("--fast", args.fast), ("--resume", args.resume), ("--force-features", args.force_features)):
        if on:
            exp.append(flag)
    if args.db_path:
        exp += ["--db-path", args.db_path]
    data_mode = "synthetic_smoke_test" if stub else args.data_mode
    if data_mode:
        exp += ["--data-mode", data_mode]
    report = [py, _script("generate_report.py"), "--artifacts-root", root, "--config", cfgp, "--experiment-id", exp_id]
    if args.db_path:
        report += ["--db-path", args.db_path]
    stages = {
        "prepare_data": [[py, _script("prepare_data.py"), "--config", cfgp]],
        "build_features": [base + ["--stage", "base", *base_flags]],
        "train_transformer": [[py, _script("train_transformer.py"), "--config", cfgp, "--artifacts-dir", root]]
                             + [base + ["--stage", "fusion", *var_flags]],
        "run_experiments": [exp],
        "generate_report": [report],
    }
    return {"stages": stages, "variants": variants, "skipped_variants": skipped, "experiment_id": exp_id,
            "stub_sentiment": stub, "sentiment_available": sent_ok, "data_mode": data_mode}


def select_stages(from_stage: str | None, only_stage: str | None) -> list[str]:
    if only_stage and from_stage:
        raise PipelineError("use either --from-stage or --only-stage, not both")
    if only_stage:
        return [only_stage]
    if from_stage:
        return STAGES[STAGES.index(from_stage):]
    return list(STAGES)


def _write_status(root: str, status: dict) -> None:
    path = os.path.join(root, "logs", "pipeline_status.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(status, f, indent=2)


def run_plan(cfg, args) -> int:
    """Execute (or, with dry_run, only print) the plan. Returns the process exit code."""
    plan = build_plan(cfg, args)
    chosen = select_stages(args.from_stage, args.only_stage)
    for v, why in plan["skipped_variants"].items():
        logger.warning("Variant %s will be N/A: %s", v, why)
    if plan["stub_sentiment"]:
        logger.warning("--stub-sentiment: sentiment features are a synthetic stub; results are a SOFTWARE SMOKE TEST ONLY.")
    env = dict(os.environ)
    if args.db_path:
        env["MARKET_MAKER_DB_PATH"] = args.db_path
    env["MARKET_MAKER_ARTIFACTS_ROOT"] = args.artifacts_root
    if args.dry_run:
        print(f"Plan (dry run, nothing executed): stages {chosen}")
        for st in chosen:
            for cmd in plan["stages"][st]:
                print(f"[{st}] " + " ".join(cmd))
        if plan["skipped_variants"]:
            print("Skipped variants:", json.dumps(plan["skipped_variants"]))
        return 0
    status = {"config": args.config, "started_utc": datetime.now(timezone.utc).isoformat(), "fast": bool(args.fast),
              "stub_sentiment": plan["stub_sentiment"], "sentiment_available": plan["sentiment_available"],
              "skipped_variants": plan["skipped_variants"], "variants": plan["variants"],
              "experiment_id": plan["experiment_id"], "data_mode": plan["data_mode"], "stages": {}}
    _write_status(args.artifacts_root, status)
    for st in chosen:
        t0 = time.time()
        for cmd in plan["stages"][st]:
            logger.info("Stage %s: %s", st, " ".join(cmd))
            res = subprocess.run(cmd, env=env)
            if res.returncode != 0:
                status["stages"][st] = {"status": "failed", "exit_code": res.returncode, "seconds": time.time() - t0}
                status["finished_utc"] = datetime.now(timezone.utc).isoformat()
                _write_status(args.artifacts_root, status)
                logger.error("Stage '%s' FAILED with exit code %d after %.1fs. Command: %s. Fix the error above, then re-run "
                             "with --from-stage %s.", st, res.returncode, time.time() - t0, " ".join(cmd), st)
                return res.returncode or 1
        status["stages"][st] = {"status": "ok", "seconds": round(time.time() - t0, 2)}
        logger.info("Stage %s finished in %.1fs", st, time.time() - t0)
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_status(args.artifacts_root, status)
    logger.info("run_all completed stages %s. Timings: %s", chosen,
                {k: v["seconds"] for k, v in status["stages"].items()})
    return 0
