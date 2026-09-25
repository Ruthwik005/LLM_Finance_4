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
