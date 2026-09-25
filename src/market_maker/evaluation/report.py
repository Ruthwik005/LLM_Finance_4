from __future__ import annotations
import json
import os

SECTIONS = [
    "Title", "Abstract", "Problem Statement", "Base Paper Summary",
    "Corrections to Initial Project Assumptions", "Proposed System", "Architecture",
    "Data Description", "FinBERT Methodology", "Transformer Methodology",
    "Market-Making Environment", "SAC Methodology", "Reward Design", "Experimental Setup",
    "Baselines and Ablations", "Results", "PnL Chart", "Inventory Chart",
    "Adverse-Selection Chart", "Discussion", "Limitations", "Future Scope",
    "Reproducibility Instructions", "References",
]

MISSING = "Not run / insufficient data"


def _load_runs(runs_root: str) -> dict:
    runs = {}
    if not os.path.isdir(runs_root):
        return runs
    for run_id in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, run_id)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.isdir(run_dir) and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                runs[run_id] = json.load(f)
            metadata_path = os.path.join(run_dir, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
            runs[run_id]["_metadata"] = metadata
    return runs


def generate_report(runs_root: str, output_path: str, template_sections: list[str] | None = None) -> str:
    sections = template_sections or SECTIONS
    runs = _load_runs(runs_root)

    lines = ["# LLM-Enhanced Financial Market Making — Final Project Report", ""]

    data_modes = {metrics.get("_metadata", {}).get("data_mode", "mode_b") for metrics in runs.values()}
    has_synthetic = "synthetic_smoke_test" in data_modes
    has_real = bool(data_modes - {"synthetic_smoke_test"})
    if has_synthetic:
        lines.append("> SOFTWARE SMOKE TEST ONLY - NOT A FINANCIAL PERFORMANCE RESULT.")
    if has_real or not runs:
        lines.append("> Data mode: Mode B — real crypto LOB, FNSPID news validated separately (not causally matched).")
    lines.append("")

    for section in sections:
        lines.append(f"## {section}")
        if section == "Results":
            if not runs:
                lines.append(MISSING)
            else:
                lines.append("| Run | Mean PnL | MAP |")
                lines.append("|---|---:|---:|")
                for run_id, metrics in runs.items():
                    mean_pnl = metrics.get("mean_pnl", MISSING)
                    map_value = metrics.get("map", MISSING)
                    lines.append(f"| {run_id} | {mean_pnl} | {map_value} |")
        elif section in ("PnL Chart", "Inventory Chart", "Adverse-Selection Chart"):
            lines.append(MISSING if not runs else "See artifacts/evaluation/<run_id>/plots/")
        else:
            lines.append(MISSING)
        lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    return text
