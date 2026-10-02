"""Final project report generator (spec section 26).

Every number in the generated Markdown comes from the SQLite DB (runs, metrics, experiment_aggregates, steps) or from
run artifacts / metadata files; everything else is static methodology (`report_text.py`). Missing data is rendered as
`N/A (<reason>)` or `Not run / insufficient data`, never invented.

Sources (flags override env vars override defaults):
  DB:        --db-path  | $MARKET_MAKER_DB_PATH        | <artifacts>/market_maker.db
  Artifacts: --artifacts-root | $MARKET_MAKER_ARTIFACTS_ROOT | artifacts
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from market_maker.evaluation import report_text as T
from market_maker.evaluation.report_text import MISSING, SMOKE_BANNER, FILL_DISCLAIMER

SECTIONS = [
    "Title", "Abstract", "Problem Statement", "Base Paper Summary",
    "Corrections to Initial Project Assumptions", "Proposed System", "Architecture",
    "Data Description", "FinBERT Methodology", "Transformer Methodology",
    "Market-Making Environment", "SAC Methodology", "Reward Design", "Experimental Setup",
    "Baselines and Ablations", "Results", "PnL Chart", "Inventory Chart",
    "Adverse-Selection Chart", "Discussion", "Limitations", "Future Scope",
    "Reproducibility Instructions", "References",
]

# (metric key, column title, kind)
PRIMARY_METRICS = [
    ("mean_pnl", "EPnL", "num"), ("map", "MAP", "num"), ("pnl_over_map", "PnL/MAP", "num"),
    ("n_fills", "Fill Count", "count"), ("adverse_selection_ratio", "Adverse Selection", "num"),
    ("cancel_ratio", "Cancel Ratio", "num"), ("max_drawdown", "Max Drawdown", "num"),
]
SECONDARY_METRICS = [
    ("sharpe_like", "Sharpe-like", "num"), ("fill_rate", "Fill Rate", "num"),
    ("avg_quoted_spread", "Avg Quoted Spread (ticks)", "num"),
    ("spread_capture_pnl", "Spread-Capture PnL", "num"), ("inventory_pnl", "Inventory PnL", "num"),
    ("fees_paid", "Fees Paid", "num"),
]
UNDEFINED_REASON = {
    "pnl_over_map": "MAP = 0 or no value",
    "adverse_selection_ratio": "no fills",
    "cancel_ratio": "no placed volume",
    "fill_rate": "no placed volume",
    "avg_quoted_spread": "no quotes with volume",
    "sharpe_like": "needs >= 2 episodes with non-zero PnL spread",
}
REFERENCE_VARIANT = "SAC_LOB_ONLY"
COMPARE_LABELS = {"mean_pnl": "EPnL", "pnl_over_map": "PnL/MAP", "map": "MAP", "max_drawdown": "max drawdown",
                  "sharpe_like": "Sharpe-like"}
PALETTE = {"HEURISTIC": "#7f7f7f", "AVELLANEDA_STOIKOV": "#4d4d4d", "SAC_LOB_ONLY": "#0072B2",
           "SAC_LOB_TRANSFORMER": "#009E73", "SAC_LOB_SENTIMENT": "#E69F00", "SAC_FULL": "#CC79A7"}


def fmt_num(x, kind: str = "num") -> str:
    """Number formatting used for every table cell (tests rely on it to match DB values)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    if kind == "count":
        return f"{x:.1f}"
    ax = abs(x)
    if ax != 0 and ax < 1e-4:
        return f"{x:.3e}"
    if ax >= 1000:
        return f"{x:.1f}"
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class ReportData:
    artifacts_root: str | None
    eval_root: str
    db_path: str | None
    db_found: bool = False
    notes: list = field(default_factory=list)
    experiment_id: str | None = None
    summary: dict | None = None
    aggregates: dict = field(default_factory=dict)  # variant -> metric -> {mean, std, n_seeds}
    agg_source: str = "none"
    runs: dict = field(default_factory=dict)  # run_id -> {variant, seed, metrics, metadata, config}
    feature_meta: dict | None = None
    pipeline_status: dict | None = None
    transformer_curve: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    config_path: str | None = None

    # -- derived -----------------------------------------------------------
    @property
    def variants(self) -> list[str]:
        names = set(self.aggregates) or {r["variant"] for r in self.runs.values()}
        ordered = [v for v in T.VARIANT_ORDER if v in names]
        return ordered + sorted(names - set(ordered))

    def runs_of(self, variant: str) -> list[dict]:
        rs = [r for r in self.runs.values() if r["variant"] == variant]
        seeds = (self.summary or {}).get("seeds")
        if seeds is not None and variant not in T.BASELINE_VARIANTS:
            rs = [r for r in rs if r["seed"] in seeds]
        return sorted(rs, key=lambda r: (r["seed"] is None, r["seed"]))

    @property
    def data_modes(self) -> set:
        modes = set()
        if self.summary and self.summary.get("data_mode"):
            modes.add(self.summary["data_mode"])
        for v in self.variants:
            for r in self.runs_of(v):
                dm = r["metadata"].get("data_mode")
                if dm:
                    modes.add(dm)
        return modes

    @property
    def synthetic(self) -> bool:
        return "synthetic_smoke_test" in self.data_modes

    def metadata_value(self, key: str):
        """First non-null value of a metadata key across the selected runs (learned runs first)."""
        ordered = sorted(self.runs.values(), key=lambda r: r["variant"] in T.BASELINE_VARIANTS)
        for r in ordered:
            if r["metadata"].get(key) is not None:
                return r["metadata"][key]
        return None


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _open_db(db_path: str | None):
    if not db_path or not os.path.exists(db_path):
        return None
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)


def _tables(conn) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def load_report_data(runs_root: str | None = None, *, artifacts_root: str | None = None, db_path: str | None = None,
                     experiment_id: str | None = None, config_path: str | None = None) -> ReportData:
    artifacts_root = artifacts_root or os.environ.get("MARKET_MAKER_ARTIFACTS_ROOT")
    if runs_root is None:
        runs_root = os.path.join(artifacts_root or "artifacts", "evaluation")
        artifacts_root = artifacts_root or "artifacts"
    elif artifacts_root is None and os.path.basename(os.path.normpath(runs_root)) == "evaluation":
        artifacts_root = os.path.dirname(os.path.normpath(runs_root))
    if db_path is None:
        db_path = os.environ.get("MARKET_MAKER_DB_PATH") or (
            os.path.join(artifacts_root, "market_maker.db") if artifacts_root else None)
    d = ReportData(artifacts_root=artifacts_root, eval_root=runs_root, db_path=db_path, config_path=config_path)

    # ---- run folders (metrics.json / metadata.json / config.yaml) -----------------------------------
    if os.path.isdir(runs_root):
        for run_id in sorted(os.listdir(runs_root)):
            rd = os.path.join(runs_root, run_id)
            mp = os.path.join(rd, "metrics.json")
            if os.path.isdir(rd) and os.path.exists(mp):
                md = _read_json(os.path.join(rd, "metadata.json")) if os.path.exists(os.path.join(rd, "metadata.json")) else {}
                cfgp = os.path.join(rd, "config.yaml")
                cfg = yaml.safe_load(open(cfgp)) if os.path.exists(cfgp) else None
                variant = md.get("variant") or run_id
                d.runs[run_id] = {"run_id": run_id, "variant": variant, "seed": md.get("seed"),
                                  "metrics": _read_json(mp), "metadata": md, "config": cfg if isinstance(cfg, dict) else {}}
    else:
        d.notes.append(f"Run folder root `{runs_root}` does not exist.")

    # ---- database -------------------------------------------------------------------------------------
    conn = _open_db(db_path)
    exp_ids: list[str] = []
    if conn is None:
        d.notes.append(f"Database `{db_path}` not found." if db_path else "No database path configured.")
    else:
        d.db_found = True
        tabs = _tables(conn)
        if "runs" in tabs:
            for run_id, variant, seed in conn.execute("SELECT run_id, variant, seed FROM runs"):
                r = d.runs.setdefault(run_id, {"run_id": run_id, "variant": variant, "seed": seed,
                                               "metrics": {}, "metadata": {}, "config": {}})
                r["variant"], r["seed"] = variant, seed
                if "metrics" in tabs:
                    db_m = {n: v for n, v in conn.execute("SELECT name, value FROM metrics WHERE run_id=?", (run_id,))}
                    if db_m:
                        r["metrics"] = db_m  # DB is the source of truth for table numbers
        if "experiment_aggregates" in tabs:
            exp_ids = [r[0] for r in conn.execute(
                "SELECT experiment_id FROM experiment_aggregates GROUP BY experiment_id ORDER BY MAX(id) DESC")]
        d._conn = conn  # type: ignore[attr-defined]
        if not d.runs and not exp_ids:
            d.notes.append("The database contains no runs and no experiment aggregates.")

    # ---- experiment selection -------------------------------------------------------------------------
    summaries: dict[str, dict] = {}
    if os.path.isdir(runs_root):
        for fn in os.listdir(runs_root):
            if fn.endswith("_summary.json"):
                s = _read_json(os.path.join(runs_root, fn))
                if isinstance(s, dict) and s.get("experiment_id"):
                    summaries[s["experiment_id"]] = s
    chosen = experiment_id
    if chosen is None:
        chosen = exp_ids[0] if exp_ids else (
            max(summaries, key=lambda k: summaries[k].get("created_utc", "")) if summaries else None)
    d.experiment_id = chosen
    d.summary = summaries.get(chosen) if chosen else None
    if experiment_id and experiment_id not in exp_ids and experiment_id not in summaries:
        d.notes.append(f"Experiment `{experiment_id}` was requested but no aggregates or summary exist for it.")
    if chosen and conn is not None and chosen in exp_ids:
        for variant, metric, mean, std, n in conn.execute(
                "SELECT variant, metric, mean, std, n_seeds FROM experiment_aggregates WHERE experiment_id=?", (chosen,)):
            d.aggregates.setdefault(variant, {})[metric] = {"mean": mean, "std": std, "n_seeds": n}
        d.agg_source = "database"
    elif d.summary:
        for variant, ms in d.summary.get("aggregates", {}).items():
            d.aggregates[variant] = {m: dict(v) for m, v in ms.items()}
        d.agg_source = "summary_json"
        d.notes.append("Aggregates were read from the experiment summary file because the database has none for this experiment.")

    # ---- optional extras --------------------------------------------------------------------------------
    if artifacts_root:
        fm = os.path.join(artifacts_root, "features", "feature_meta.json")
        if os.path.exists(fm):
            d.feature_meta = _read_json(fm)
        ps = os.path.join(artifacts_root, "logs", "pipeline_status.json")
        if os.path.exists(ps):
            d.pipeline_status = _read_json(ps)
        tc = os.path.join(artifacts_root, "transformer", "training_curve.csv")
        if os.path.exists(tc):
            import csv
            with open(tc) as f:
                d.transformer_curve = list(csv.DictReader(f))

    # ---- config: prefer a learned run's config.yaml, else explicit file --------------------------------
    cfgs = sorted(d.runs.values(), key=lambda r: r["variant"] in T.BASELINE_VARIANTS)
    for r in cfgs:
        if r["config"] and "simulator" in r["config"]:
            d.config = r["config"]
            break
    if not d.config and config_path and os.path.exists(config_path):
        d.config = yaml.safe_load(open(config_path))
    return d


def _cv(d: ReportData, path: str, default: str = "N/A (not recorded)"):
    cur = d.config
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _bar_with_err(ax, d: ReportData, metric: str, title: str, ylabel: str, variants=None):
    variants = [v for v in (variants or d.variants) if metric in d.aggregates.get(v, {})]
    xs = range(len(variants))
    means = [d.aggregates[v][metric]["mean"] for v in variants]
    errs = [d.aggregates[v][metric]["std"] or 0.0 for v in variants]
    ax.bar(xs, means, yerr=errs, capsize=4, color=[PALETTE.get(v, "#999999") for v in variants])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([T.VARIANT_LABELS.get(v, v) for v in variants], rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.6)
    return variants


def _mean_curves(d: ReportData, sql_expr: str) -> dict[str, tuple]:
    """variant -> (steps, mean over seeds of the per-run average over episodes of `sql_expr`) from the steps table."""
    conn = getattr(d, "_conn", None)
    if conn is None or "steps" not in _tables(conn):
        return {}
    import numpy as np
    out = {}
    for v in d.variants:
        per_run = []
        for r in d.runs_of(v):
            rows = conn.execute(f"SELECT step, AVG({sql_expr}) FROM steps WHERE run_id=? GROUP BY step ORDER BY step",
                                (r["run_id"],)).fetchall()
            if rows:
                per_run.append(rows)
        if per_run:
            n = min(len(p) for p in per_run)
            steps = np.array([x[0] for x in per_run[0][:n]])
            vals = np.mean([[x[1] for x in p[:n]] for p in per_run], axis=0)
            out[v] = (steps, vals)
    return out


def make_figures(d: ReportData, fig_dir: str) -> dict:
    """Write PNG figures; returns {name: file name or None}. A figure is skipped (None) when its data is absent."""
    os.makedirs(fig_dir, exist_ok=True)
    plt = _plt()
    figs: dict = {}

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, name), dpi=130)
        plt.close(fig)
        return name

    if any("mean_pnl" in d.aggregates.get(v, {}) for v in d.variants):
        curves = _mean_curves(d, "mtm_pnl")
        fig, axes = plt.subplots(1, 2 if curves else 1, figsize=(11 if curves else 6, 4.2), squeeze=False)
        _bar_with_err(axes[0][0], d, "mean_pnl", "Mean episode PnL (mean +/- std over seeds)", "PnL (quote units)")
        if curves:
            ax = axes[0][1]
            for v, (steps, vals) in curves.items():
                ax.plot(steps, vals, label=T.VARIANT_LABELS.get(v, v), color=PALETTE.get(v), linewidth=1.4)
            ax.set_title("Mean cumulative mark-to-market PnL\n(avg over test episodes, then over seeds)")
            ax.set_xlabel("step in episode")
            ax.set_ylabel("PnL")
            ax.legend(fontsize=7)
        figs["pnl_comparison.png"] = _save(fig, "pnl_comparison.png")
    else:
        figs["pnl_comparison.png"] = None

    if any("map" in d.aggregates.get(v, {}) for v in d.variants):
        curves = _mean_curves(d, "ABS(inventory)")
        fig, axes = plt.subplots(1, 2 if curves else 1, figsize=(11 if curves else 6, 4.2), squeeze=False)
        ax = axes[0][0]
        _bar_with_err(ax, d, "map", "Mean absolute position (MAP)", "units")
        if curves:
            ax = axes[0][1]
            for v, (steps, vals) in curves.items():
                ax.plot(steps, vals, label=T.VARIANT_LABELS.get(v, v), color=PALETTE.get(v), linewidth=1.4)
            ax.set_title("Mean |inventory| over the episode\n(avg over test episodes, then over seeds)")
            ax.set_xlabel("step in episode")
            ax.set_ylabel("|inventory|")
            ax.legend(fontsize=7)
        figs["inventory_comparison.png"] = _save(fig, "inventory_comparison.png")
    else:
        figs["inventory_comparison.png"] = None

    if any("adverse_selection_ratio" in d.aggregates.get(v, {}) for v in d.variants):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        vs = _bar_with_err(ax, d, "adverse_selection_ratio", "Adverse-selection ratio (approximation)", "fraction of fills")
        for i, v in enumerate(vs):
            nf = d.aggregates[v].get("n_fills", {}).get("mean")
            if nf is not None:
                ax.annotate(f"fills ~{nf:.0f}", (i, d.aggregates[v]["adverse_selection_ratio"]["mean"]),
                            ha="center", va="bottom", fontsize=7, xytext=(0, 4), textcoords="offset points")
        figs["adverse_selection_comparison.png"] = _save(fig, "adverse_selection_comparison.png")
    else:
        figs["adverse_selection_comparison.png"] = None

    # learning curves: validation reward of each learned variant (mean over seeds, band = std)
    import numpy as np
    import pandas as pd
    curves = {}
    for v in d.variants:
        if v in T.BASELINE_VARIANTS:
            continue
        frames = []
        for r in d.runs_of(v):
            p = os.path.join(d.eval_root, r["run_id"], "training_curve.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                if {"timesteps", "val_reward_mean"} <= set(df.columns) and len(df):
                    frames.append(df.set_index("timesteps")["val_reward_mean"])
        if frames:
            curves[v] = pd.concat(frames, axis=1)
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for v, df in curves.items():
            m, s = df.mean(axis=1), df.std(axis=1, ddof=1) if df.shape[1] > 1 else None
            ax.plot(m.index, m.values, label=f"{T.VARIANT_LABELS.get(v, v)} (n={df.shape[1]})", color=PALETTE.get(v))
            if s is not None:
                ax.fill_between(m.index, (m - s).values, (m + s).values, color=PALETTE.get(v), alpha=0.15)
        ax.set_title("Validation reward during training (mean over seeds, band = std)")
        ax.set_xlabel("training timesteps")
        ax.set_ylabel("mean validation episode reward")
        ax.legend(fontsize=7)
        figs["learning_curves.png"] = _save(fig, "learning_curves.png")
    else:
        figs["learning_curves.png"] = None
    return figs


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _na_reason_variant(d: ReportData, variant: str) -> str:
    skipped = ((d.pipeline_status or {}).get("skipped_variants") or {})
    if variant in skipped:
        return skipped[variant]
    if d.experiment_id:
        return f"variant not run in experiment `{d.experiment_id}`"
    return "variant not run"


def _cell(d: ReportData, variant: str, metric: str, kind: str, reasons: list) -> str:
    if variant not in d.aggregates:
        why = _na_reason_variant(d, variant)
        reasons.append(f"- {T.VARIANT_LABELS.get(variant, variant)} (all metrics): {why}.")
        return "N/A (not run)"
    a = d.aggregates[variant].get(metric)
    if a is None:
        why = UNDEFINED_REASON.get(metric, "metric not produced for this run")
        reasons.append(f"- {T.VARIANT_LABELS.get(variant, variant)} / {metric}: metric undefined in every run ({why}).")
        return "N/A (undefined)"
    n = a["n_seeds"]
    if variant in T.BASELINE_VARIANTS:
        return fmt_num(a["mean"], kind)
    if a["std"] is None:
        return f"{fmt_num(a['mean'], kind)} (n=1, std N/A)"
    return f"{fmt_num(a['mean'], kind)} ± {fmt_num(a['std'], kind)} (n={n})"


def _table(d: ReportData, metrics, rows_variants, reasons) -> list[str]:
    lines = ["| Model | " + " | ".join(m[1] for m in metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for v in rows_variants:
        cells = [_cell(d, v, m, k, reasons) for m, _, k in metrics]
        lines.append(f"| {T.VARIANT_LABELS.get(v, v)} | " + " | ".join(cells) + " |")
    return lines


def _per_run_table(d: ReportData) -> list[str]:
    """Fallback when no experiment aggregates exist: one row per run, single values."""
    cols = [("mean_pnl", "EPnL", "num"), ("map", "MAP", "num"), ("pnl_over_map", "PnL/MAP", "num"),
            ("n_fills", "Fill Count", "count"), ("adverse_selection_ratio", "Adverse Selection", "num"),
            ("cancel_ratio", "Cancel Ratio", "num"), ("max_drawdown", "Max Drawdown", "num")]
    lines = ["| Run | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for rid, r in d.runs.items():
        cells = []
        for m, _, k in cols:
            x = r["metrics"].get(m)
            cells.append(fmt_num(x, k) if x is not None else "N/A (undefined or not recorded)")
        lines.append(f"| {rid} | " + " | ".join(cells) + " |")
    return lines


def _banner(d: ReportData) -> list[str]:
    out = []
    if d.synthetic:
        out.append(f"> {SMOKE_BANNER}.")
        out.append("")
    modes = d.data_modes - {"synthetic_smoke_test"}
    if modes & {"mode_b"}:
        out.append("> Data mode: Mode B — real crypto LOB, FNSPID news validated separately (not causally matched).")
        out.append("")
    out.append(f"> {FILL_DISCLAIMER}")
    out.append("")
    return out


def _sec_title(d):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    modes = ", ".join(sorted(d.data_modes)) or "not recorded"
    return [f"**LLM-Enhanced Financial Market Making using FinBERT, Transformer Networks and Soft Actor-Critic Reinforcement Learning**", "",
            f"- Report generated: {ts}",
            f"- Experiment id: `{d.experiment_id or 'N/A (no experiment found)'}`",
            f"- Data mode(s) recorded in run metadata: {modes}",
            f"- Code version (git commit recorded in run metadata): `{d.metadata_value('git_commit') or 'N/A (not recorded)'}`",
            "- Research and backtesting only. No live trading, no broker connectivity."]


def _sec_abstract(d):
    if not d.runs and not d.aggregates:
        return ["This report documents the design of a news-aware, multi-price-level market-making research system (FinBERT sentiment, "
                "Transformer temporal encoder, Soft Actor-Critic agent) evaluated by historical replay. "
                f"No experiment results were found for this report ({MISSING}); see Section 16 for what is missing."]
    vs = [T.VARIANT_LABELS.get(v, v) for v in d.variants]
    seeds = (d.summary or {}).get("seeds")
    sd = f" over {len(seeds)} seed(s)" if seeds else ""
    out = ["This report documents a news-aware, multi-price-level market-making research system that extends the structure of the IMM "
           "paper (IJCAI 2024) with FinBERT sentiment features, a frozen Transformer temporal encoder and a Soft Actor-Critic (SAC) agent, "
           "evaluated by historical replay with an approximate snapshot fill simulator.",
           "",
           f"The experiment `{d.experiment_id or 'N/A'}` compared {len(vs)} model(s) ({', '.join(vs)}){sd} on fixed test windows. "
           "Section 16 reports the measured values (mean ± sample std across seeds for learned variants, single deterministic evaluations "
           "for baselines) and Section 20 discusses them together with exploratory p-values and effect sizes."]
    if d.synthetic:
        out += ["", f"**{SMOKE_BANNER}.** The numbers in this report come from synthetic data and prove only that the software pipeline runs "
                "end to end; they carry no financial meaning."]
    return out


def _sec_problem(d):
    return ["A market maker continuously quotes buy and sell prices and earns the spread when both sides trade, but it carries inventory risk "
            "(the price can move against its position) and suffers adverse selection (fills tend to happen just before unfavorable price moves). "
            "Quotes must be chosen from the limit order book (LOB) state and the agent's own inventory, repeatedly and under uncertainty, "
            "which makes the task a sequential decision problem.",
            "",
            "The project is designed to evaluate whether (a) a text-derived sentiment channel (FinBERT) and (b) a temporal Transformer representation "
            "of the recent market, when added to a SAC agent that quotes two price levels per side, change the measured PnL and risk metrics "
            "relative to an LOB-only SAC agent and to two non-learned baselines. We make no assumption in advance about the direction or size of any effect."]


def _sec_data(d):
    out = []
    if not d.runs:
        out.append(f"N/A (no run metadata found, so the dataset cannot be described). {MISSING}.")
    else:
        modes = ", ".join(sorted(d.data_modes)) or "not recorded in run metadata"
        out.append(f"- Data mode recorded in run metadata: **{modes}**.")
        sym, depth, tick = _cv(d, "data.symbol"), _cv(d, "data.depth"), _cv(d, "data.tick_size")
        out.append(f"- Symbol: `{sym}`; LOB depth: {depth}; tick size: {tick}; LOB file (config): `{_cv(d, 'data.lob_path')}`; news file (config): `{_cv(d, 'data.news_path')}`.")
        ranges = d.metadata_value("date_ranges")
        if ranges:
            out += ["", "| Split | Start | End | Rows |", "|---|---|---|---:|"]
            for sp in ("train", "validation", "test"):
                if sp in ranges:
                    r = ranges[sp]
                    out.append(f"| {sp} | {r['start']} | {r['end']} | {r['rows']} |")
            out += ["", "Splits are chronological by trading day (no shuffling); split boundaries are stored in each run's `metadata.json`."]
        else:
            out.append("- Split date ranges: N/A (not recorded in run metadata).")
        fp = d.metadata_value("dataset_fingerprints")
        if fp:
            out.append(f"- Dataset fingerprints (SHA-256): LOB `{fp.get('lob_sha256')}`; news `{fp.get('news_sha256')}`.")
        else:
            out.append("- Dataset fingerprints: N/A (not recorded in run metadata).")
    if d.feature_meta:
        rep = d.feature_meta.get("lob_validation_report")
        if rep:
            out.append(f"- LOB validation report (rows removed): {rep}.")
        sm = d.feature_meta.get("sentiment", {})
        if sm.get("source"):
            out.append(f"- News items scored: {sm.get('n_news', 'N/A')} (sentiment source: `{sm['source']}`).")
    else:
        out.append("- Feature metadata (`feature_meta.json`) not found: validation counts and news counts are N/A.")
    out += ["", "**Data modes.** The code supports three modes. Mode A: matched real LOB and real news. **Mode B (used for real runs): real crypto "
            "LOB (self-collected Binance order book, or public Binance data) plus real equity news from FNSPID, validated separately and not "
            "causally matched** — the two datasets come from different assets and periods, so the sentiment pipeline is validated on its own "
            "and no claim is made that news improved trading on the crypto book. Mode C: deterministic synthetic data, used only as a software smoke test."]
    if d.synthetic:
        out += ["", f"**Synthetic data notice.** {SMOKE_BANNER}. The runs in this report used synthetic LOB and/or synthetic stub sentiment; "
                "stub sentiment scores are random and carry no information."]
    if d.pipeline_status and d.pipeline_status.get("stub_sentiment"):
        out.append("- The pipeline was run with `--stub-sentiment`: sentiment features come from a deterministic synthetic stub, not FinBERT.")
    return out


def _sec_finbert(d):
    lb = _cv(d, "sentiment.lookback_minutes", "N/A")
    text = T.FINBERT_METHOD.format(model_name=_cv(d, "sentiment.model_name", "ProsusAI/finbert"), lookbacks=lb,
                                   max_age=_cv(d, "sentiment.max_age_minutes", "N/A"))
    out = text.split("\n")
    if d.feature_meta and d.feature_meta.get("sentiment", {}).get("source"):
        src = d.feature_meta["sentiment"]["source"]
        note = ("**Measured in this build:** the sentiment source was `synthetic_stub` (random scores, smoke test only; not FinBERT)."
                if src == "synthetic_stub" else f"**Measured in this build:** sentiment source `{src}`.")
        out += ["", note]
    else:
        out += ["", "Sentiment source of this build: N/A (no sentiment features recorded; sentiment variants cannot be evaluated)."]
    return out


def _sec_transformer(d):
    t = lambda k, dflt="N/A": _cv(d, f"transformer.{k}", dflt)  # noqa: E731
    text = T.TRANSFORMER_METHOD.format(
        seq=t("sequence_length"), d_model=t("d_model"), layers=t("num_layers"), heads=t("nhead"), ff=t("dim_feedforward"),
        dropout=t("dropout"), horizons=t("horizons_minutes"), thr=_cv(d, "features.label_threshold_bps", "N/A"),
        cw=t("class_weights"))
    out = text.split("\n")
    cur = d.transformer_curve
    if cur:
        best = min(cur, key=lambda r: float(r["val_loss"]))
        out += ["", f"**Measured Transformer training ({len(cur)} epoch(s) logged):** best epoch {best['epoch']} with validation loss "
                f"{float(best['val_loss']):.4f}. Validation accuracy versus the majority-class rate at that epoch:"]
        for k in sorted(c for c in best if c.startswith("val_acc_")):
            h = k[len("val_acc_"):]
            maj = best.get(f"val_majority_acc_{h}")
            out.append(f"- horizon {h}: accuracy {float(best[k]):.4f}" + (f" versus majority-class {float(maj):.4f}" if maj is not None else ""))
        out.append("Accuracy at or near the majority-class rate means the classifier has learned little beyond the class prior.")
    else:
        out += ["", "Measured Transformer training: N/A (no `artifacts/transformer/training_curve.csv` found; the Transformer variants cannot have been run)."]
    return out


def _sec_env(d):
    g = lambda p: _cv(d, p, "N/A")  # noqa: E731
    text = T.ENV_METHOD.format(
        max_off=g("simulator.max_mid_offset_ticks"), min_sp=g("simulator.min_spread_ticks"), max_sp=g("simulator.max_spread_ticks"),
        tick=g("data.tick_size"), vol=g("simulator.total_volume_per_side"), ref=g("simulator.reference_price_mode"),
        latency=g("simulator.latency_steps"), fee=g("simulator.fee_per_unit"), rebate=g("simulator.rebate_per_unit"),
        max_inv=g("simulator.max_inventory"), disclaimer=FILL_DISCLAIMER)
    return text.split("\n") + ["", f"Episode length: {g('simulator.episode_minutes')} minutes at {g('simulator.action_interval_ms')} ms per step."]


def _sec_sac(d):
    g = lambda p: _cv(d, p, "N/A")  # noqa: E731
    steps = d.metadata_value("total_timesteps")
    fast = d.metadata_value("fast_mode")
    out = ["Soft Actor-Critic (Stable-Baselines3, `MlpPolicy`) is an off-policy maximum-entropy actor-critic method for continuous actions. "
           "It is used here for the four-dimensional quote action; no claim is made that it is better than TD3 (TD3 was not run).",
           "",
           f"- Hyperparameters (config): learning rate {g('sac.learning_rate')}, buffer {g('sac.buffer_size')}, learning starts {g('sac.learning_starts')}, "
           f"batch size {g('sac.batch_size')}, gamma {g('sac.gamma')}, tau {g('sac.tau')}, entropy coefficient `{g('sac.ent_coef')}`, "
           f"network {g('sac.net_arch')}, device `{g('sac.device')}`.",
           f"- Training budget actually used (run metadata): {steps if steps is not None else 'N/A (not recorded)'} timesteps per run"
           + (" (fast smoke-test budget)." if fast else "."),
           f"- **Validation-based checkpoint selection:** every {g('sac.eval_freq')} steps the deterministic policy is evaluated on fixed validation windows; "
           "the checkpoint with the best mean validation episode reward is saved as `best_model.zip` and is the only one used for test evaluation. "
           "The test split is never used for selection or tuning.",
           f"- Seeds: {(d.summary or {}).get('seeds', 'N/A (no experiment summary)')}. Every learned variant uses the same split, windows, action limits, fees, reward and training budget."]
    return out


def _sec_reward(d):
    g = lambda p: _cv(d, p, "N/A")  # noqa: E731
    rtype = g("reward.type")
    pen = (f"`inventory_penalty = {g('reward.inventory_penalty')} * |inventory|` applied only when |inventory| > {g('reward.inventory_threshold')} (truncated, IMM-style)"
           if rtype == "truncated_inventory" else
           f"`inventory_penalty = {g('reward.inventory_penalty')} * inventory^2` (quadratic)" if rtype == "quadratic"
           else f"reward type `{rtype}`")
    return ["```text", "reward_t = scale * (delta_mark_to_market_pnl - inventory_penalty - cancellation_penalty)", "```",
            f"- Reward type: `{rtype}`; {pen}.",
            f"- Cancellation penalty: `{g('reward.cancellation_penalty')} * canceled_volume`; scale: {g('reward.scale')}.",
            f"- Fees ({g('simulator.fee_per_unit')} per unit) and rebates ({g('simulator.rebate_per_unit')} per unit) enter through the accounting, hence through the mark-to-market PnL; no artificial compensation bonus is added.",
            f"- The config field `reward.adverse_selection_penalty` ({g('reward.adverse_selection_penalty')}) is **not applied** by the implemented reward function.",
            "- Reward components are computed separately each step; reward is not the same thing as financial performance, so the tables report PnL, MAP and the other metrics."]


def _sec_setup(d):
    s = d.summary or {}
    if not d.runs:
        return [f"N/A (no runs found). {MISSING}."]
    out = [f"- Experiment id: `{d.experiment_id or 'N/A'}`; variants: {', '.join(d.variants) or 'N/A'}.",
           f"- Seeds: {s.get('seeds', 'N/A (no summary file)')}; fast (smoke-test) mode: {s.get('fast_mode', d.metadata_value('fast_mode'))}.",
           f"- Training budget per SAC run: {d.metadata_value('total_timesteps') if d.metadata_value('total_timesteps') is not None else 'N/A'} timesteps; "
           f"validation evaluation every {d.metadata_value('eval_freq') if d.metadata_value('eval_freq') is not None else 'N/A (not recorded in run metadata)'} steps.",
           f"- Episode length: {s.get('episode_steps', _cv(d, 'simulator.episode_minutes', 'N/A'))} "
           + ("steps." if s.get("episode_steps") else "minutes."),
           f"- Test windows: {s.get('n_test_windows', 'N/A')} fixed, evenly spaced (start indices {s.get('test_windows', 'N/A')}); validation windows: {s.get('n_val_windows', 'N/A')}.",
           f"- Seed spread is the sample standard deviation across seeds (ddof = 1); `n` in the tables is the number of seeds that defined the metric.",
           f"- Adverse selection: a fill counts as adverse if the mid moves against it by more than {_cv(d, 'evaluation.adverse_threshold_ticks', 'N/A')} tick(s) within "
           f"{_cv(d, 'evaluation.adverse_horizon_steps', 'N/A')} steps (approximate; not claimed equal to the IMM paper's metric)."]
    timings = s.get("timings") or {}
    if timings:
        tot = sum(t["train_seconds"] for t in timings.values())
        sps = [t["steps_per_second"] for t in timings.values()]
        out.append(f"- Measured SAC training time: {tot:.0f} s over {len(timings)} run(s) (training throughput {min(sps):.0f}-{max(sps):.0f} steps/s).")
    return out


def _sec_baselines(d):
    out = ["| Model | Description | Status |", "|---|---|---|"]
    for v in T.VARIANT_ORDER:
        if v in d.aggregates:
            n = d.aggregates[v].get("mean_pnl", {}).get("n_seeds") or next(iter(d.aggregates[v].values()))["n_seeds"]
            st = f"ran (n={n})"
        else:
            st = f"N/A: {_na_reason_variant(d, v)}"
        out.append(f"| {T.VARIANT_LABELS[v]} | {T.VARIANT_DESCRIPTIONS[v]} | {st} |")
    out.append("| TD3 (optional) | Same observation/action space with TD3 | N/A: not implemented (optional extension) |")
    out.append("| IMM-style imitation / behavior cloning (optional) | Expert demonstrations + BC loss | N/A: not implemented (optional extension) |")
    out += ["", "All learned variants share split, windows, action limits, fees, reward definition and training budget. "
            "Ablations differ only in which feature blocks are concatenated to the (identical) LOB block."]
    return out


def _sec_results(d, fig_rel):
    if not d.aggregates and not d.runs:
        reason = "; ".join(d.notes) or "no runs or aggregates in the database or run folders"
        return [f"{MISSING} ({reason})."]
    out = []
    reasons: list = []
    if d.aggregates:
        out += [f"Experiment `{d.experiment_id}` (source: {d.agg_source}). Learned variants: mean ± sample std over seeds with `n` = number of seeds; "
                "baselines: a single deterministic evaluation (no spread). Metrics are computed over the fixed test windows.", ""]
        rows = [v for v in T.VARIANT_ORDER if v in d.aggregates or v in (d.pipeline_status or {}).get("skipped_variants", {})
                or v in ("SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL", "HEURISTIC")]
        rows += [v for v in d.variants if v not in rows]
        out += ["**Primary table (spec section 43)**", ""] + _table(d, PRIMARY_METRICS, rows, reasons)
        out += ["", "**Secondary metrics**", ""] + _table(d, SECONDARY_METRICS, rows, reasons)
        out += ["", "Definitions: EPnL = mean episode PnL; MAP = mean absolute position (units); Fill Count = total fills over all test windows; "
                "Cancel Ratio = canceled / placed volume; Fill Rate = filled / placed volume; Avg Quoted Spread in ticks; Max Drawdown = worst "
                "per-episode drawdown of the mark-to-market curve; Spread-Capture and Inventory PnL are totals over all test episodes "
                "(Inventory PnL = total PnL - spread capture + fees - rebates)."]
        if reasons:
            out += ["", "**N/A reasons**", ""] + sorted(set(reasons))
    else:
        out += ["No experiment aggregates were found; single-run values from the run folders / database are shown (n = 1 per run).", ""]
        out += _per_run_table(d)
    out += ["", "**Learning curves**", ""]
    out += [f"![Validation reward during training]({fig_rel['learning_curves.png']})"] if fig_rel.get("learning_curves.png") else \
        ["N/A (no `training_curve.csv` with validation rewards found for the learned runs)."]
    return out


def _chart_section(d, fig_rel, name, caption, reason):
    if fig_rel.get(name):
        return [f"![{caption}]({fig_rel[name]})", "", caption + "."]
    return [f"{MISSING}: {reason}"]


def _sec_discussion(d):
    if not d.aggregates:
        return [f"{MISSING}: there are no experiment aggregates to discuss."]
    out = ["Everything below restates measured values; it is not a claim about markets. The Welch tests are exploratory (small n, uncorrected for "
           "the number of comparisons, same fixed test windows for all seeds)."]
    if d.synthetic:
        out += ["", f"**{SMOKE_BANNER}.** Results on synthetic data carry no financial meaning."]
    learned = [v for v in d.variants if v not in T.BASELINE_VARIANTS]
    ns = [a["mean_pnl"]["n_seeds"] for v, a in d.aggregates.items() if v in learned and "mean_pnl" in a]
    if ns:
        n = min(ns)
        out += ["", f"Number of seeds per learned variant: {min(ns)}" + (f"-{max(ns)}" if max(ns) != min(ns) else "") +
                (". With so few seeds the tests have low power: a small difference cannot be distinguished from seed-to-seed variation, "
                 "and a large one may still be a fluke." if n < 10 else ".")]
    comps = (d.summary or {}).get(f"comparisons_vs_{REFERENCE_VARIANT}") or {}
    if comps:
        n_cmp = sum(len(m) for m in comps.values())
        out += ["", f"**Learned variants versus {T.VARIANT_LABELS[REFERENCE_VARIANT]}** ({n_cmp} comparisons, no multiple-testing correction):", ""]
        for v, ms in comps.items():
            for m, c in ms.items():
                label = COMPARE_LABELS.get(m, m)
                if c.get("mean_diff") is None:
                    continue
                p, cd = c.get("p_value"), c.get("cohens_d")
                head = (f"- {T.VARIANT_LABELS.get(v, v)} minus {T.VARIANT_LABELS[REFERENCE_VARIANT]}, {label}: mean difference {c['mean_diff']:+.4f} "
                        f"(n = {c['n_a']} vs {c['n_b']}")
                if p is None:
                    out.append(head + "); no test possible with fewer than 2 seeds per group.")
                else:
                    verdict = ("within seed-to-seed noise at this sample size" if p >= 0.05 else
                               "p below 0.05, but exploratory only (small n, uncorrected multiple comparisons)")
                    cds = f", Cohen's d {cd:+.2f}" if cd is not None else ""
                    out.append(head + f"; Welch p = {p:.3f}{cds}): {verdict}.")
    base = [v for v in T.BASELINE_VARIANTS if v in d.aggregates and "mean_pnl" in d.aggregates[v]]
    if base and learned:
        out += ["", "**Learned variants versus the baselines (EPnL).** Baselines are single deterministic evaluations without a spread estimate, so no test is applied:", ""]
        for v in learned:
            a = d.aggregates.get(v, {}).get("mean_pnl")
            if not a:
                continue
            sd = f"{a['std']:.4f}" if a["std"] is not None else "N/A"
            parts = [f"{T.VARIANT_LABELS[b]} {d.aggregates[b]['mean_pnl']['mean']:.4f} (learned minus baseline {a['mean'] - d.aggregates[b]['mean_pnl']['mean']:+.4f})" for b in base]
            out.append(f"- {T.VARIANT_LABELS.get(v, v)}: EPnL {a['mean']:.4f} (std across seeds {sd}); " + "; ".join(parts) + ".")
    # diagnostics from measured values
    diag = []
    max_inv = _cv(d, "simulator.max_inventory", None)
    for v in d.variants:
        a = d.aggregates[v]
        nf = a.get("n_fills", {}).get("mean")
        if nf is not None and nf == 0:
            diag.append(f"{T.VARIANT_LABELS.get(v, v)} recorded zero fills in the test windows, so its PnL reflects no trading.")
        cr = a.get("cancel_ratio", {}).get("mean")
        if cr is not None and cr > 0.9:
            diag.append(f"{T.VARIANT_LABELS.get(v, v)} canceled {cr:.0%} of the volume it placed (high quote churn).")
        mi = a.get("max_abs_inventory", {}).get("mean")
        if mi is not None and isinstance(max_inv, (int, float)) and mi >= 0.95 * max_inv:
            diag.append(f"{T.VARIANT_LABELS.get(v, v)} reached an average maximum |inventory| of {mi:.1f}, at the inventory limit ({max_inv}).")
        adv, pnl = a.get("adverse_selection_ratio", {}).get("mean"), a.get("mean_pnl", {}).get("mean")
        if adv is not None and adv > 0.5 and pnl is not None and pnl < 0:
            diag.append(f"{T.VARIANT_LABELS.get(v, v)} has negative EPnL ({pnl:.4f}) and an adverse-selection ratio of {adv:.2f}.")
    if diag:
        out += ["", "**Diagnostics (measured):**", ""] + [f"- {x}" for x in diag]
    out += ["", "No statement of improvement is made unless the numbers above support it with an appropriate test; where differences are inside the "
            "spread across seeds they should be read as no detectable difference, not as equivalence."]
    return out


def _sec_limitations(d):
    out = [f"- {x}" for x in T.LIMITATIONS + T.LIMITATIONS_EXTRA]
    ns = [a["mean_pnl"]["n_seeds"] for a in d.aggregates.values() if "mean_pnl" in a]
    if ns:
        out.append(f"- This run used at most {max(ns)} seed(s) per variant.")
    out.append(f"- {FILL_DISCLAIMER}")
    return out


def _sec_repro(d, fig_rel):
    cfgp = d.config_path or "<config.yaml>"
    s = d.summary or {}
    seeds = " ".join(str(x) for x in s.get("seeds", [])) or "<seeds>"
    fast = " --fast" if s.get("fast_mode") else ""
    stub = " --stub-sentiment" if (d.pipeline_status or {}).get("stub_sentiment") else ""
    exp = d.experiment_id or "<experiment-id>"
    out = ["```bash", "git clone <repo-url> && cd <repo>", "python3.11 -m venv .venv && source .venv/bin/activate",
           "pip install -r requirements.txt && pip install -e .",
           f"python scripts/prepare_data.py --config {cfgp}",
           f"python scripts/build_features.py --config {cfgp} --stage base{stub}",
           f"python scripts/train_transformer.py --config {cfgp}",
           f"python scripts/build_features.py --config {cfgp} --stage fusion{stub}",
           f"python scripts/run_experiments.py --config {cfgp} --experiment-id {exp} --seeds {seeds}{fast}",
           "python scripts/generate_report.py", "```", "",
           "Or run all stages with `python scripts/run_all.py --config <config.yaml>` (see README).", ""]
    fields = [("Git commit", d.metadata_value("git_commit")), ("Config hash", d.metadata_value("config_hash")),
              ("Feature hash", d.metadata_value("feature_hash")), ("Python", d.metadata_value("python_version")),
              ("Project seed", d.metadata_value("seed")), ("Package versions", d.metadata_value("package_versions")),
              ("Simulator mode", d.metadata_value("simulator_mode")), ("Reference price mode", d.metadata_value("reference_price_mode"))]
    for k, v in fields:
        out.append(f"- {k}: `{v}`" if v is not None else f"- {k}: N/A (not recorded in run metadata)")
    out.append("- Results are reproducible as far as the framework and platform permit; bit-for-bit determinism across hardware is not promised.")
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def generate_report(runs_root: str | None, output_path: str, template_sections: list[str] | None = None, *,
                    db_path: str | None = None, artifacts_root: str | None = None, experiment_id: str | None = None,
                    config_path: str | None = None) -> str:
    d = load_report_data(runs_root, artifacts_root=artifacts_root, db_path=db_path, experiment_id=experiment_id,
                         config_path=config_path)
    sections = template_sections or SECTIONS
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    figs = make_figures(d, fig_dir)
    fig_rel = {k: (os.path.relpath(os.path.join(fig_dir, v), out_dir).replace(os.sep, "/") if v else None) for k, v in figs.items()}

    builders = {
        "Title": lambda: _sec_title(d), "Abstract": lambda: _sec_abstract(d), "Problem Statement": lambda: _sec_problem(d),
        "Base Paper Summary": lambda: T.BASE_PAPER.split("\n"),
        "Corrections to Initial Project Assumptions": lambda: (
            ["| # | Initial assumption | Correction | Handling in this project |", "|---|---|---|---|"] +
            [f"| {i} | {a} | {b} | {c} |" for i, (a, b, c) in enumerate(T.CORRECTIONS, 1)]),
        "Proposed System": lambda: [
            "A news-aware, multi-price-level market-making research system: (1) LOB features; (2) FinBERT sentiment features aligned without look-ahead; "
            "(3) a frozen Transformer temporal encoder pretrained on multi-horizon price-direction classification; (4) feature fusion into a flat observation "
            "together with private account state; (5) a SAC agent whose continuous action defines two bid and two ask levels; (6) a historical-replay "
            "environment with inventory limits, fees, latency and an approximate fill simulator; (7) baselines (heuristic, Avellaneda-Stoikov) and "
            "ablations (LOB only, +Transformer, +FinBERT, full) evaluated over several seeds on fixed test windows; (8) a read-only API, a React dashboard and this report.",
            "", "The title keeps the phrase \"LLM-enhanced\" for continuity; technically the language-model component is the pretrained FinBERT sentiment classifier (see Section 9)."],
        "Architecture": lambda: T.ARCH_MERMAID.split("\n") + [""] + T.ARCH_NOTES.split("\n"),
        "Data Description": lambda: _sec_data(d), "FinBERT Methodology": lambda: _sec_finbert(d),
        "Transformer Methodology": lambda: _sec_transformer(d), "Market-Making Environment": lambda: _sec_env(d),
        "SAC Methodology": lambda: _sec_sac(d), "Reward Design": lambda: _sec_reward(d), "Experimental Setup": lambda: _sec_setup(d),
        "Baselines and Ablations": lambda: _sec_baselines(d), "Results": lambda: _sec_results(d, fig_rel),
        "PnL Chart": lambda: _chart_section(d, fig_rel, "pnl_comparison.png", "Figure 1: PnL comparison across variants",
                                            "no mean PnL aggregates available"),
        "Inventory Chart": lambda: _chart_section(d, fig_rel, "inventory_comparison.png", "Figure 2: Inventory (MAP) comparison across variants",
                                                  "no MAP aggregates available"),
        "Adverse-Selection Chart": lambda: _chart_section(d, fig_rel, "adverse_selection_comparison.png",
                                                          "Figure 3: Adverse-selection ratio across variants (approximation)",
                                                          "no adverse-selection ratio defined for any variant (no fills or no aggregates)"),
        "Discussion": lambda: _sec_discussion(d), "Limitations": lambda: _sec_limitations(d),
        "Future Scope": lambda: [f"- {x}" for x in T.FUTURE_SCOPE],
        "Reproducibility Instructions": lambda: _sec_repro(d, fig_rel),
        "References": lambda: [f"{i}. {r}" for i, r in enumerate(T.REFERENCES, 1)],
    }
    lines = ["# LLM-Enhanced Financial Market Making — Final Project Report", ""] + _banner(d)
    if d.notes:
        lines += ["Data availability notes: " + " ".join(d.notes), ""]
    for i, section in enumerate(sections, 1):
        lines.append(f"## {i}. {section}")
        lines.append("")
        lines += builders[section]() if section in builders else [MISSING]
        lines.append("")
    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    conn = getattr(d, "_conn", None)
    if conn is not None:
        conn.close()
    return text
