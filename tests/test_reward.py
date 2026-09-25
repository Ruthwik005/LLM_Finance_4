from market_maker.env.reward import compute_reward


def test_positive_mtm_delta_increases_reward():
    reward_pos, _ = compute_reward(delta_mtm_pnl=1.0, inventory=0.0, canceled_volume=0.0,
                                    reward_type="truncated_inventory", inventory_threshold=40,
                                    inventory_penalty=0.001, cancellation_penalty=0.0001, scale=1.0)
    reward_neg, _ = compute_reward(delta_mtm_pnl=-1.0, inventory=0.0, canceled_volume=0.0,
                                    reward_type="truncated_inventory", inventory_threshold=40,
                                    inventory_penalty=0.001, cancellation_penalty=0.0001, scale=1.0)
    assert reward_pos > reward_neg


def test_truncated_inventory_penalty_zero_below_threshold():
    reward, components = compute_reward(delta_mtm_pnl=0.0, inventory=10.0, canceled_volume=0.0,
                                         reward_type="truncated_inventory", inventory_threshold=40,
                                         inventory_penalty=0.5, cancellation_penalty=0.0, scale=1.0)
    assert components["inventory_penalty"] == 0.0


def test_truncated_inventory_penalty_activates_above_threshold():
    reward, components = compute_reward(delta_mtm_pnl=0.0, inventory=50.0, canceled_volume=0.0,
                                         reward_type="truncated_inventory", inventory_threshold=40,
                                         inventory_penalty=0.5, cancellation_penalty=0.0, scale=1.0)
    assert components["inventory_penalty"] == 0.5 * 50.0


def test_quadratic_inventory_penalty():
    reward, components = compute_reward(delta_mtm_pnl=0.0, inventory=3.0, canceled_volume=0.0,
                                         reward_type="quadratic", inventory_threshold=40,
                                         inventory_penalty=0.1, cancellation_penalty=0.0, scale=1.0)
    assert components["inventory_penalty"] == 0.1 * 9.0
