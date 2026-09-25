from __future__ import annotations


def compute_reward(
    delta_mtm_pnl: float,
    inventory: float,
    canceled_volume: float,
    reward_type: str,
    inventory_threshold: float,
    inventory_penalty: float,
    cancellation_penalty: float,
    scale: float,
) -> tuple[float, dict]:
    if reward_type == "truncated_inventory":
        inv_pen = inventory_penalty * abs(inventory) if abs(inventory) > inventory_threshold else 0.0
    elif reward_type == "quadratic":
        inv_pen = inventory_penalty * (inventory ** 2)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    cancel_pen = cancellation_penalty * canceled_volume
    components = {"pnl": delta_mtm_pnl, "inventory_penalty": inv_pen, "cancellation_penalty": cancel_pen}
    reward = scale * (delta_mtm_pnl - inv_pen - cancel_pen)
    return reward, components
