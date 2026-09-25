from __future__ import annotations
import json
import os
import csv
import yaml
import numpy as np


def run_backtest(env, agent, n_episodes: int, is_heuristic: bool, max_inventory: float) -> dict:
    episode_pnls, inventories, all_fills, all_orders = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        inventory_trace = []
        ep_fills = []
        cumulative_pnl = 0.0
        while not done:
            if is_heuristic:
                action = agent.predict(obs, inventory=env.account.inventory, max_inventory=max_inventory)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            inventory_trace.append(info["inventory"])
            ep_fills.extend(info["fills"])
            cumulative_pnl += info["reward_components"]["pnl"]
            done = terminated or truncated
        episode_pnls.append(cumulative_pnl)
        inventories.append(np.array(inventory_trace))
        all_fills.append(ep_fills)
        all_orders.append([])

    return {"episode_pnls": episode_pnls, "inventories": inventories, "fills": all_fills, "orders": all_orders}


def write_run_artifacts(run_dir: str, config: dict, metadata: dict, backtest_result: dict, metrics: dict) -> None:
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
        writer.writerow(["episode", "note"])

    with open(os.path.join(run_dir, "training_curve.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward"])

    with open(os.path.join(run_dir, "model_path.txt"), "w") as f:
        f.write(metadata.get("model_path", "N/A"))

    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write("# Run report placeholder — see artifacts/reports/FINAL_PROJECT_REPORT.md for the full report.\n")
