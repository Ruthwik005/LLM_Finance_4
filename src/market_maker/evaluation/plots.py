from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_plots(plots_dir: str, result: dict, metrics: dict, config: dict | None = None) -> list[str]:
    """Write equity_curve, episode_pnl, inventory, adverse_selection PNGs. Returns written paths."""
    os.makedirs(plots_dir, exist_ok=True)
    steps = result.get("steps") or []
    pnls = result["episode_pnls"]
    paths = []

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, ep in enumerate(steps):
        ax.plot([s["mtm_pnl"] for s in ep], alpha=0.7, label=f"ep {i}" if i < 8 else None)
    ax.set_xlabel("step"); ax.set_ylabel("mark-to-market PnL"); ax.set_title("Equity curve per episode")
    if steps:
        ax.legend(fontsize=6)
    paths.append(os.path.join(plots_dir, "equity_curve.png")); _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(len(pnls)), pnls, color=["tab:green" if p >= 0 else "tab:red" for p in pnls])
    ax.set_xlabel("episode"); ax.set_ylabel("PnL"); ax.set_title("Episode PnL")
    paths.append(os.path.join(plots_dir, "episode_pnl.png")); _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, inv in enumerate(result["inventories"]):
        ax.plot(inv, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("step"); ax.set_ylabel("inventory"); ax.set_title("Inventory over time")
    paths.append(os.path.join(plots_dir, "inventory.png")); _save(fig, paths[-1])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    adv = metrics.get("adverse_selection_ratio")
    if adv is None:
        ax.text(0.5, 0.5, "no evaluable fills", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(["adverse", "not adverse"], [adv, 1 - adv], color=["tab:red", "tab:gray"])
        ax.set_ylim(0, 1)
    ax.set_title("Adverse-selection share of fills")
    paths.append(os.path.join(plots_dir, "adverse_selection.png")); _save(fig, paths[-1])
    return paths
