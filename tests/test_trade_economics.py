"""Tests for trade_economics — the single source of truth for financial math."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_economics import calculate, TradeEconomics


class TestTradeEconomics:
    """Golden tests: these exact numbers must never change."""

    def test_yes_basic(self):
        """YES at 30¢: 1000 tokens → $300 cost, $700 profit, 2.33x"""
        e = calculate(size=1000, price=0.30, outcome="Yes")
        assert e.is_no == False
        assert e.cost == pytest.approx(300.0)
        assert e.potential_profit == pytest.approx(700.0)
        assert e.pnl_multiplier == pytest.approx(7 / 3, rel=1e-3)
        assert e.effective_odds == pytest.approx(0.30)
        assert e.raw_price == 0.30

    def test_no_basic(self):
        """NO at 90¢ YES price: 1000 tokens → $100 cost, $900 profit, 9x"""
        e = calculate(size=1000, price=0.90, outcome="No")
        assert e.is_no == True
        assert e.cost == pytest.approx(100.0)
        assert e.potential_profit == pytest.approx(900.0)
        assert e.pnl_multiplier == pytest.approx(9.0)
        assert e.roi_percent == pytest.approx(900.0)
        assert e.effective_odds == pytest.approx(0.10)

    def test_no_roi_never_negative(self):
        """The historical -100% bug: NO trades must always have positive ROI."""
        for price in [0.10, 0.50, 0.70, 0.90, 0.95, 0.99]:
            e = calculate(size=100, price=price, outcome="No")
            assert e.roi_percent > 0, f"NO at YES price {price} got ROI {e.roi_percent}"
            assert e.cost > 0

    def test_yes_roi_never_negative(self):
        for price in [0.01, 0.10, 0.50, 0.70, 0.90]:
            e = calculate(size=100, price=price, outcome="Yes")
            assert e.roi_percent > 0, f"YES at {price} got ROI {e.roi_percent}"

    def test_50_50_market(self):
        """50/50 → cost = $500, profit = $500, 1x ROI for both sides"""
        yes = calculate(size=1000, price=0.50, outcome="Yes")
        no = calculate(size=1000, price=0.50, outcome="No")
        assert yes.cost == pytest.approx(500.0)
        assert no.cost == pytest.approx(500.0)
        assert yes.pnl_multiplier == pytest.approx(1.0)
        assert no.pnl_multiplier == pytest.approx(1.0)

    def test_extreme_yes(self):
        """YES at 1¢: 1000 tokens → $10 cost, $990 profit, 99x"""
        e = calculate(size=1000, price=0.01, outcome="Yes")
        assert e.cost == pytest.approx(10.0)
        assert e.pnl_multiplier == pytest.approx(99.0)

    def test_extreme_no(self):
        """NO at 1¢ YES price (99¢ NO): 1000 tokens → $990 cost, $10 profit"""
        e = calculate(size=1000, price=0.01, outcome="No")
        assert e.cost == pytest.approx(990.0)
        assert e.potential_profit == pytest.approx(10.0, rel=1e-2)

    def test_cost_symmetry(self):
        """YES cost + NO cost = size (they're complementary)"""
        size = 1000
        for price in [0.10, 0.30, 0.50, 0.70, 0.90]:
            yes = calculate(size=size, price=price, outcome="Yes")
            no = calculate(size=size, price=price, outcome="No")
            assert yes.cost + no.cost == pytest.approx(size)

    def test_default_outcome_is_yes(self):
        e = calculate(size=100, price=0.50)
        assert e.is_no == False

    def test_none_outcome_is_yes(self):
        e = calculate(size=100, price=0.50, outcome=None)
        assert e.is_no == False

    def test_zero_price(self):
        """Edge case: price = 0 should not crash"""
        e = calculate(size=100, price=0.0, outcome="Yes")
        assert e.cost == 0.0

    def test_dataclass_immutable_fields(self):
        """Verify all fields are populated"""
        e = calculate(size=500, price=0.40, outcome="No")
        assert isinstance(e, TradeEconomics)
        assert e.outcome == "No"
        assert e.tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
