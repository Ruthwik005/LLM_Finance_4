from __future__ import annotations
import json
import os
import csv
import yaml
import numpy as np


def run_backtest(env, agent, n_episodes: int, is_heuristic: bool, max_inventory: float) -> dict:
    """Run n_episodes. ``is_heuristic=True`` means the agent exposes
    ``predict(obs, inventory=, max_inventory=)`` (heuristic and Avellaneda-Stoikov); otherwise SB3 ``predict``."""
    episode_pnls, inventories, all_fills, all_orders, all_steps = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        inventory_trace, ep_fills, ep_steps, ep_orders = [], [], [], []
        cumulative_pnl = 0.0
        while not done:
            if is_heuristic:
                action = agent.predict(obs, inventory=env.account.inventory, max_inventory=max_inventory)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            inventory_trace.append(info["inventory"])
            ep_fills.extend(info["fills"])
            log = info["step_log"]
            ep_steps.append(log)
            ep_orders.append({"t": log["t"], "placed_volume": log["placed_volume"],
                              "canceled_volume": log["canceled_volume"],
                              "outstanding_volume": float(sum(log["bid_volumes"]) + sum(log["ask_volumes"]))})
            cumulative_pnl += info["reward_components"]["pnl"]
            done = terminated or truncated
        episode_pnls.append(cumulative_pnl)
        inventories.append(np.array(inventory_trace))
        all_fills.append(ep_fills)
        all_orders.append(ep_orders)
        all_steps.append(ep_steps)

    return {"episode_pnls": episode_pnls, "inventories": inventories, "fills": all_fills,
            "orders": all_orders, "steps": all_steps}


STEP_COLUMNS = ["episode", "t", "timestamp", "mid", "next_mid", "bid_price_1", "bid_price_2", "ask_price_1",
                "ask_price_2", "bid_vol_1", "bid_vol_2", "ask_vol_1", "ask_vol_2", "inventory", "cash",
                "mtm_pnl", "n_fills", "placed_volume", "canceled_volume"]


def flatten_step(episode: int, s: dict) -> list:
    return [episode, s["t"], s["timestamp"], s["mid"], s.get("next_mid"), *s["bid_prices"], *s["ask_prices"],
            *s["bid_volumes"], *s["ask_volumes"], s["inventory"], s["cash"], s["mtm_pnl"], len(s["fills"]),
            s["placed_volume"], s["canceled_volume"]]


def write_run_artifacts(run_dir: str, config: dict, metadata: dict, backtest_result: dict, metrics: dict,
                        training_curve: list[tuple[float, float]] | None = None) -> None:
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(run_dir, "episodes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "pnl"])
        for i, pnl in enumerate(backtest_result["episode_pnls"]):
            writer.writerow([i, pnl])

    with open(os.path.join(run_dir, "fills.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "side", "price", "volume"])
        for i, ep_fills in enumerate(backtest_result["fills"]):
            for fill in ep_fills:
                writer.writerow([i, fill["side"], fill["price"], fill["volume"]])

    with open(os.path.join(run_dir, "orders.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "t", "placed_volume", "canceled_volume", "outstanding_volume"])
        for i, ep_orders in enumerate(backtest_result.get("orders", [])):
            for o in ep_orders:
                writer.writerow([i, o["t"], o["placed_volume"], o["canceled_volume"], o["outstanding_volume"]])

    with open(os.path.join(run_dir, "quotes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(STEP_COLUMNS)
        for i, ep_steps in enumerate(backtest_result.get("steps", [])):
            for s in ep_steps:
                writer.writerow(flatten_step(i, s))

    with open(os.path.join(run_dir, "training_curve.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward"])
        for step, reward in (training_curve or []):
            writer.writerow([step, reward])

    try:
        from market_maker.evaluation.plots import write_plots
        write_plots(os.path.join(run_dir, "plots"), backtest_result, metrics, config)
    except Exception as exc:  # plots are auxiliary; never fail an evaluation over them
        import logging
        logging.getLogger(__name__).warning("plot generation failed: %s", exc)

    with open(os.path.join(run_dir, "model_path.txt"), "w") as f:
        f.write(metadata.get("model_path", "N/A"))

    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write("# Run report placeholder — see artifacts/reports/FINAL_PROJECT_REPORT.md for the full report.\n")
