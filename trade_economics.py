"""
Trade Economics — Single source of truth for position cost, PnL, and ROI.

All financial calculations live here. No other module should compute
cost, profit, or ROI independently.

Polymarket mechanics:
- API always returns YES token price (0..1)
- Each token pays $1.00 if correct outcome, $0 otherwise
- YES cost = size * price
- NO cost  = size * (1 - price)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeEconomics:
    """Immutable snapshot of trade financial properties."""
    outcome: str           # "Yes" or "No"
    is_no: bool
    raw_price: float       # YES token price from API (always YES side)
    effective_odds: float  # probability the trader is betting on
    cost: float            # actual $ spent
    tokens: float          # number of tokens bought
    potential_profit: float  # $ if trade wins
    pnl_multiplier: float  # profit / cost
    roi_percent: float     # pnl_multiplier * 100


def calculate(
    size: float,
    price: float,
    outcome: str = "Yes",
) -> TradeEconomics:
    """
    Calculate all trade economics from raw API data.
    
    Args:
        size: number of tokens (from trade['size'])
        price: YES token price (from trade['price']), always 0..1
        outcome: "Yes" or "No" (from trade['outcome'])
    
    Returns:
        TradeEconomics with all computed fields.
    """
    is_no = bool(outcome) and outcome.lower() == "no"
    
    if is_no:
        cost = size * (1 - price)
        effective_odds = 1 - price
    else:
        cost = size * price
        effective_odds = price
    
    # Tokens bought = cost / token_price
    token_price = (1 - price) if is_no else price
    tokens = cost / token_price if token_price > 0 else 0
    
    # Each token pays $1 if correct → payout = tokens * $1
    payout = tokens
    potential_profit = payout - cost
    
    # Multiplier and ROI
    pnl_multiplier = potential_profit / cost if cost > 0 else 0
    roi_percent = pnl_multiplier * 100
    
    return TradeEconomics(
        outcome=outcome or "Yes",
        is_no=is_no,
        raw_price=price,
        effective_odds=effective_odds,
        cost=cost,
        tokens=tokens,
        potential_profit=potential_profit,
        pnl_multiplier=pnl_multiplier,
        roi_percent=roi_percent,
    )
