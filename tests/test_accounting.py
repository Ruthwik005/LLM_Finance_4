from market_maker.env.accounting import AccountState


def test_buy_fill_increases_inventory_and_decreases_cash():
    acct = AccountState(cash=1000.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
    acct.apply_fill(side="bid", price=100.0, volume=2.0, fee_per_unit=0.0)
    assert acct.inventory == 2.0
    assert acct.cash == 1000.0 - 200.0
    assert acct.filled_bid_volume == 2.0


def test_sell_fill_decreases_inventory_and_increases_cash():
    acct = AccountState(cash=1000.0, inventory=5.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
    acct.apply_fill(side="ask", price=100.0, volume=2.0, fee_per_unit=0.0)
    assert acct.inventory == 3.0
    assert acct.cash == 1000.0 + 200.0
    assert acct.filled_ask_volume == 2.0


def test_fees_are_deducted_from_cash():
    acct = AccountState(cash=1000.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
    acct.apply_fill(side="bid", price=100.0, volume=1.0, fee_per_unit=0.5)
    assert acct.cash == 1000.0 - 100.0 - 0.5
    assert acct.fees == 0.5


def test_mark_to_market_uses_current_midprice():
    acct = AccountState(cash=800.0, inventory=2.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=2.0, filled_ask_volume=0.0, turnover=0.0)
    total_pnl = acct.mark_to_market(midprice=110.0)
    assert total_pnl == (800.0 + 2.0 * 110.0) - 800.0
