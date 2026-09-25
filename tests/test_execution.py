from market_maker.env.execution import Order, SnapshotExecutionSimulator


def test_bid_fills_when_ask_moves_through_price():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="bid", price=100.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=99.5, next_best_ask=99.9,  # ask moved below our bid price
                      next_trade_price=None, next_trade_side=None, next_trade_size=None)
    assert len(fills) == 1
    assert fills[0]["side"] == "bid"
    assert fills[0]["volume"] == 1.0


def test_bid_does_not_fill_when_market_never_reaches_price():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="bid", price=90.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=99.5, next_best_ask=100.1,
                      next_trade_price=100.0, next_trade_side="sell", next_trade_size=0.5)
    assert fills == []  # Scenario E: no trade reaches the passive quote


def test_partial_fill_capped_by_available_trade_volume():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="ask", price=100.0, volume=5.0)]
    fills = sim.step(orders, next_best_bid=100.1, next_best_ask=100.2,  # bid moved above our ask
                      next_trade_price=100.0, next_trade_side="buy", next_trade_size=2.0)
    assert len(fills) == 1
    assert fills[0]["volume"] == 2.0  # capped by available trade volume, not full 5.0


def test_ask_fills_when_bid_moves_through_price():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="ask", price=100.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=100.5, next_best_ask=100.6,
                      next_trade_price=None, next_trade_side=None, next_trade_size=None)
    assert len(fills) == 1
    assert fills[0]["side"] == "ask"


def test_no_overfill_beyond_outstanding_order_volume():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="bid", price=100.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=99.0, next_best_ask=99.5,
                      next_trade_price=99.0, next_trade_side="sell", next_trade_size=10.0)
    assert fills[0]["volume"] == 1.0  # never exceeds outstanding order volume


def test_trade_volume_not_double_counted_across_multiple_orders():
    """Verify that trade volume is conserved across multiple orders in one step.

    When a single reported trade could satisfy multiple resting orders' trade-based
    fill conditions, the trade volume should be shared (allocated) rather than
    allowing each order to independently claim the full trade size.
    """
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    # Two bid orders at different prices, both satisfying trade_reached condition
    orders = [
        Order(side="bid", price=100.5, volume=3.0),
        Order(side="bid", price=100.0, volume=3.0),
    ]
    # Single sell trade of size 4.0 that satisfies both orders
    fills = sim.step(
        orders,
        next_best_bid=99.5,
        next_best_ask=100.2,
        next_trade_price=99.5,  # sell trade below both bid prices
        next_trade_side="sell",
        next_trade_size=4.0,  # total available trade volume
    )

    # Both orders should fill, but total should not exceed 4.0
    assert len(fills) == 2
    total_filled = sum(fill["volume"] for fill in fills)
    assert total_filled == 4.0, f"Expected total fill of 4.0, got {total_filled}"
    # First order in list should be prioritized: gets min(3.0, 4.0) = 3.0
    assert fills[0]["volume"] == 3.0
    # Second order gets remaining: min(3.0, 1.0) = 1.0
    assert fills[1]["volume"] == 1.0
