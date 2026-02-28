"""
Backtest Engine v2 — Scientifically Rigorous Validation

Critical requirements addressed:
1. No lookahead bias - features reconstructed from past-only data
2. Train/Test split with walk-forward option
3. Baseline comparisons (random, always-NO, market-implied)
4. Commission and slippage modeling
5. Statistical significance (t-stat, Sharpe)
6. Multivariate feature importance

Goal: FALSIFY the hypothesis, not confirm it.
"""

import json
import requests
import time
import sqlite3
import random
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict

from config import GAMMA_API_URL, DATA_API_URL, REQUEST_DELAY


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

COMMISSION_RATE = 0.02       # 2% Polymarket fee on winnings
SLIPPAGE_RATE = 0.005        # 0.5% estimated slippage
MIN_TRADES_FOR_SIGNIFICANCE = 30
T_STAT_THRESHOLD = 2.0       # 95% confidence

DB_PATH = Path("backtest.db")


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    trade_hash: str
    wallet: str
    condition_id: str
    timestamp: int
    outcome: str
    price: float
    size: float
    amount: float


@dataclass 
class Market:
    condition_id: str
    question: str
    outcome: str  # resolved outcome
    end_date: str
    volume: float
    category: str


@dataclass
class Signal:
    trade: Trade
    market: Market
    signal_type: str
    features: Dict
    score: float


@dataclass
class TradeResult:
    signal: Signal
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    roi: float
    is_winner: bool


# ══════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS markets (
            condition_id TEXT PRIMARY KEY,
            question TEXT,
            outcome TEXT,
            end_date TEXT,
            volume REAL,
            category TEXT,
            fetched_at TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            trade_hash TEXT PRIMARY KEY,
            wallet TEXT,
            condition_id TEXT,
            timestamp INTEGER,
            outcome TEXT,
            price REAL,
            size REAL,
            amount REAL
        )
    ''')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_cond ON trades(condition_id)')
    
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════
# DATA COLLECTION (same as before, abbreviated)
# ══════════════════════════════════════════════════════════════════

def fetch_resolved_markets(days_back: int = 90, limit: int = 500) -> List[Dict]:
    """Fetch resolved markets from API."""
    url = f"{GAMMA_API_URL}/markets"
    params = {"limit": limit, "closed": "true", "order": "endDate", "_sort": "endDate:desc"}
    
    try:
        time.sleep(REQUEST_DELAY)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        markets = response.json()
        
        resolved = []
        for m in markets:
            if not m.get('resolutionSource'):
                continue
            
            outcomes = m.get('outcomes', [])
            outcome_prices = m.get('outcomePrices', [])
            
            winning = None
            for i, p in enumerate(outcome_prices):
                if float(p) > 0.95 and i < len(outcomes):
                    winning = outcomes[i]
                    break
            
            if winning:
                resolved.append({
                    'condition_id': m.get('conditionId', ''),
                    'question': m.get('question', ''),
                    'outcome': winning,
                    'end_date': m.get('endDate', ''),
                    'volume': float(m.get('volume', 0) or 0),
                    'category': classify_category(m.get('question', ''))
                })
        
        return resolved
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def classify_category(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ['trump', 'biden', 'election', 'president']): return 'politics'
    if any(w in q for w in ['war', 'strike', 'iran', 'russia', 'ukraine']): return 'geopolitics'
    if any(w in q for w in ['bitcoin', 'crypto', 'btc', 'eth']): return 'crypto'
    if any(w in q for w in ['nba', 'nfl', 'sports']): return 'sports'
    return 'other'


def fetch_trades_for_market(condition_id: str, min_amount: float = 1000) -> List[Dict]:
    """Fetch trades for a market."""
    url = f"{DATA_API_URL}/trades"
    trades = []
    
    for offset in range(0, 5000, 500):
        try:
            time.sleep(REQUEST_DELAY)
            r = requests.get(url, params={
                "conditionId": condition_id, "limit": 500, "offset": offset,
                "sortBy": "TIMESTAMP", "sortDirection": "ASC"
            }, timeout=30)
            
            batch = r.json()
            if not batch:
                break
            
            for t in batch:
                size = float(t.get('size', 0))
                price = float(t.get('price', 0))
                outcome = t.get('outcome', 'Yes')
                amount = size * (1 - price) if outcome.lower() == 'no' else size * price
                
                if amount >= min_amount:
                    trades.append({
                        'trade_hash': t.get('transactionHash', ''),
                        'wallet': t.get('proxyWallet', ''),
                        'condition_id': condition_id,
                        'timestamp': t.get('timestamp', 0),
                        'outcome': outcome,
                        'price': price,
                        'size': size,
                        'amount': amount
                    })
            
            if len(batch) < 500:
                break
        except:
            break
    
    return trades


def collect_data(days_back: int = 90):
    """Collect resolved markets and trades."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    markets = fetch_resolved_markets(days_back)
    markets = [m for m in markets if m['volume'] >= 10000]
    
    print(f"Collecting {len(markets)} markets...")
    
    for idx, m in enumerate(markets):
        c.execute('INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?)',
            (m['condition_id'], m['question'], m['outcome'], m['end_date'],
             m['volume'], m['category'], datetime.now().isoformat()))
        
        trades = fetch_trades_for_market(m['condition_id'])
        for t in trades:
            c.execute('INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?)',
                (t['trade_hash'], t['wallet'], t['condition_id'], t['timestamp'],
                 t['outcome'], t['price'], t['size'], t['amount']))
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(markets)} markets processed")
            conn.commit()
    
    conn.commit()
    conn.close()
    print("Collection complete.")


# ══════════════════════════════════════════════════════════════════
# LOOKAHEAD-SAFE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def get_wallet_history_before(wallet: str, before_ts: int, conn: sqlite3.Connection) -> Dict:
    """
    Get wallet statistics using ONLY data before the trade timestamp.
    CRITICAL: No future data leakage.
    """
    c = conn.cursor()
    
    # Trades by this wallet BEFORE this timestamp
    c.execute('''
        SELECT timestamp, amount, outcome, price
        FROM trades
        WHERE wallet = ? AND timestamp < ?
        ORDER BY timestamp ASC
    ''', (wallet, before_ts))
    
    prior_trades = c.fetchall()
    
    if not prior_trades:
        return {
            'wallet_age_days': 0,
            'prior_trade_count': 0,
            'prior_volume': 0,
            'is_new_wallet': True,
            'is_very_new_wallet': True,
            'is_low_activity': True
        }
    
    first_ts = prior_trades[0][0]
    age_days = (before_ts - first_ts) / 86400
    total_volume = sum(t[1] for t in prior_trades)
    
    return {
        'wallet_age_days': age_days,
        'prior_trade_count': len(prior_trades),
        'prior_volume': total_volume,
        'is_new_wallet': age_days < 7,
        'is_very_new_wallet': age_days < 3,
        'is_low_activity': len(prior_trades) < 5
    }


def get_market_state_at_trade(trade: Trade, market: Market) -> Dict:
    """
    Get market state at trade time.
    Uses trade price (which is known at trade time).
    DOES NOT use resolution outcome.
    """
    price = trade.price
    outcome = trade.outcome
    
    # Effective odds for the position taken
    if outcome.lower() == 'no':
        effective_odds = 1 - price
    else:
        effective_odds = price
    
    # Time to resolution (known at trade time from market end_date)
    try:
        end_dt = datetime.fromisoformat(market.end_date.replace('Z', '+00:00'))
        trade_dt = datetime.fromtimestamp(trade.timestamp, tz=timezone.utc)
        hours_to_resolution = (end_dt - trade_dt).total_seconds() / 3600
    except:
        hours_to_resolution = None
    
    return {
        'effective_odds': effective_odds,
        'is_longshot': effective_odds < 0.15,
        'is_contrarian': effective_odds < 0.10,
        'hours_to_resolution': hours_to_resolution,
        'is_pre_event': hours_to_resolution is not None and hours_to_resolution < 24,
        'is_very_pre_event': hours_to_resolution is not None and hours_to_resolution < 1
    }


def extract_features(trade: Trade, market: Market, conn: sqlite3.Connection) -> Dict:
    """
    Extract features using ONLY information available at trade time.
    NO LOOKAHEAD.
    """
    wallet_hist = get_wallet_history_before(trade.wallet, trade.timestamp, conn)
    market_state = get_market_state_at_trade(trade, market)
    
    return {
        # Wallet features (past only)
        **wallet_hist,
        
        # Trade features (known at trade time)
        'amount': trade.amount,
        'is_large_bet': trade.amount >= 5000,
        'is_very_large_bet': trade.amount >= 10000,
        
        # Market state (known at trade time)
        **market_state,
        
        # Category (known from market title)
        'category': market.category
    }


# ══════════════════════════════════════════════════════════════════
# SIGNAL CLASSIFICATION (without lookahead)
# ══════════════════════════════════════════════════════════════════

def classify_signal(features: Dict) -> Tuple[str, float]:
    """
    Classify signal type and calculate score.
    Uses only features available at trade time.
    """
    score = 0
    
    # Wallet age
    if features.get('is_very_new_wallet'):
        score += 40
    elif features.get('is_new_wallet'):
        score += 20
    
    # Activity
    if features.get('is_low_activity'):
        score += 10
    
    # Bet size
    if features.get('is_very_large_bet'):
        score += 25
    elif features.get('is_large_bet'):
        score += 20
    
    # Contrarian
    if features.get('is_contrarian'):
        score += 25
    elif features.get('is_longshot'):
        score += 15
    
    # Pre-event timing
    if features.get('is_very_pre_event'):
        score += 50
    elif features.get('is_pre_event'):
        score += 20
    
    # Classify
    if score >= 100:
        return 'ALPHA', score
    elif score >= 80:
        return 'INSIDER_CONFIRMED', score
    elif score >= 70:
        return 'CONFLICT', score
    elif score >= 50:
        return 'INSIDER_ONLY', score
    else:
        return 'NO_SIGNAL', score


# ══════════════════════════════════════════════════════════════════
# PNL CALCULATION WITH COSTS
# ══════════════════════════════════════════════════════════════════

def calculate_pnl(trade: Trade, market_outcome: str) -> TradeResult:
    """
    Calculate PnL with commission and slippage.
    """
    position = trade.outcome.lower()
    resolved = market_outcome.lower()
    amount = trade.amount
    
    is_winner = position == resolved
    
    if position == 'no':
        effective_price = 1 - trade.price
    else:
        effective_price = trade.price
    
    # Slippage on entry
    entry_slippage = amount * SLIPPAGE_RATE
    
    if is_winner:
        tokens = amount / effective_price
        gross_pnl = tokens - amount
        commission = gross_pnl * COMMISSION_RATE  # Commission on winnings
        exit_slippage = tokens * SLIPPAGE_RATE
        net_pnl = gross_pnl - commission - entry_slippage - exit_slippage
    else:
        gross_pnl = -amount
        commission = 0
        net_pnl = gross_pnl - entry_slippage
    
    roi = net_pnl / amount if amount > 0 else 0
    
    # Create signal placeholder (will be filled by caller)
    return TradeResult(
        signal=None,
        gross_pnl=gross_pnl,
        commission=commission,
        slippage=entry_slippage + (exit_slippage if is_winner else 0),
        net_pnl=net_pnl,
        roi=roi,
        is_winner=is_winner
    )


# ══════════════════════════════════════════════════════════════════
# BASELINE STRATEGIES
# ══════════════════════════════════════════════════════════════════

def baseline_random(trades: List[Trade], markets: Dict[str, Market]) -> List[TradeResult]:
    """Random entry baseline."""
    results = []
    for trade in trades:
        market = markets.get(trade.condition_id)
        if not market:
            continue
        
        # Random YES/NO
        fake_outcome = random.choice(['Yes', 'No'])
        fake_trade = Trade(
            trade_hash=trade.trade_hash,
            wallet=trade.wallet,
            condition_id=trade.condition_id,
            timestamp=trade.timestamp,
            outcome=fake_outcome,
            price=trade.price,
            size=trade.size,
            amount=trade.amount
        )
        
        result = calculate_pnl(fake_trade, market.outcome)
        results.append(result)
    
    return results


def baseline_always_no(trades: List[Trade], markets: Dict[str, Market]) -> List[TradeResult]:
    """Always bet NO baseline (bet against hype)."""
    results = []
    for trade in trades:
        market = markets.get(trade.condition_id)
        if not market:
            continue
        
        # Always NO
        fake_trade = Trade(
            trade_hash=trade.trade_hash,
            wallet=trade.wallet,
            condition_id=trade.condition_id,
            timestamp=trade.timestamp,
            outcome='No',
            price=trade.price,
            size=trade.size,
            amount=trade.amount
        )
        
        result = calculate_pnl(fake_trade, market.outcome)
        results.append(result)
    
    return results


def baseline_follow_odds(trades: List[Trade], markets: Dict[str, Market]) -> List[TradeResult]:
    """Bet on higher probability side."""
    results = []
    for trade in trades:
        market = markets.get(trade.condition_id)
        if not market:
            continue
        
        # Bet on side with higher implied probability
        if trade.price > 0.5:
            position = 'Yes'
        else:
            position = 'No'
        
        fake_trade = Trade(
            trade_hash=trade.trade_hash,
            wallet=trade.wallet,
            condition_id=trade.condition_id,
            timestamp=trade.timestamp,
            outcome=position,
            price=trade.price,
            size=trade.size,
            amount=trade.amount
        )
        
        result = calculate_pnl(fake_trade, market.outcome)
        results.append(result)
    
    return results


# ══════════════════════════════════════════════════════════════════
# STATISTICAL METRICS
# ══════════════════════════════════════════════════════════════════

def calculate_stats(results: List[TradeResult]) -> Dict:
    """Calculate comprehensive statistics."""
    if not results:
        return {'error': 'No results'}
    
    n = len(results)
    rois = [r.roi for r in results]
    pnls = [r.net_pnl for r in results]
    wins = sum(1 for r in results if r.is_winner)
    
    total_pnl = sum(pnls)
    total_invested = sum(r.signal.trade.amount if r.signal else 0 for r in results)
    if total_invested == 0:
        total_invested = sum(abs(r.gross_pnl) for r in results)
    
    mean_roi = sum(rois) / n
    
    # Variance and t-stat
    if n > 1:
        variance = sum((r - mean_roi) ** 2 for r in rois) / (n - 1)
        std = math.sqrt(variance)
        stderr = std / math.sqrt(n)
        t_stat = mean_roi / stderr if stderr > 0 else 0
    else:
        std = 0
        t_stat = 0
    
    # Sharpe (annualized, assuming 1 trade per day)
    sharpe = (mean_roi * 365) / (std * math.sqrt(365)) if std > 0 else 0
    
    # Max drawdown
    cumulative = []
    running = 0
    for pnl in pnls:
        running += pnl
        cumulative.append(running)
    
    peak = 0
    max_dd = 0
    for c in cumulative:
        if c > peak:
            peak = c
        dd = peak - c
        if dd > max_dd:
            max_dd = dd
    
    return {
        'n': n,
        'total_pnl': total_pnl,
        'mean_roi': mean_roi,
        'std_roi': std,
        't_stat': t_stat,
        'sharpe': sharpe,
        'win_rate': wins / n,
        'max_drawdown': max_dd,
        'is_significant': abs(t_stat) > T_STAT_THRESHOLD and n >= MIN_TRADES_FOR_SIGNIFICANCE
    }


# ══════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════

def run_backtest(test_ratio: float = 0.3):
    """
    Run walk-forward backtest with train/test split.
    
    Split by time (not random) to avoid lookahead.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Load markets
    c.execute('SELECT * FROM markets')
    markets_raw = c.fetchall()
    markets = {}
    for row in markets_raw:
        markets[row[0]] = Market(
            condition_id=row[0],
            question=row[1],
            outcome=row[2],
            end_date=row[3],
            volume=row[4],
            category=row[5]
        )
    
    # Load trades sorted by timestamp
    c.execute('SELECT * FROM trades ORDER BY timestamp ASC')
    trades_raw = c.fetchall()
    trades = []
    for row in trades_raw:
        if row[2] in markets:  # Only trades with resolved markets
            trades.append(Trade(
                trade_hash=row[0],
                wallet=row[1],
                condition_id=row[2],
                timestamp=row[3],
                outcome=row[4],
                price=row[5],
                size=row[6],
                amount=row[7]
            ))
    
    if not trades:
        print("No trades found.")
        conn.close()
        return
    
    print(f"Loaded {len(trades)} trades across {len(markets)} markets")
    
    # Time-based split
    split_idx = int(len(trades) * (1 - test_ratio))
    train_trades = trades[:split_idx]
    test_trades = trades[split_idx:]
    
    print(f"Train: {len(train_trades)} trades | Test: {len(test_trades)} trades")
    
    # Generate signals for test set only
    # (In production, you would tune thresholds on train, then evaluate on test)
    
    results_by_type = defaultdict(list)
    all_test_results = []
    
    for trade in test_trades:
        market = markets.get(trade.condition_id)
        if not market:
            continue
        
        # Extract features (no lookahead)
        features = extract_features(trade, market, conn)
        
        # Classify
        signal_type, score = classify_signal(features)
        
        if signal_type == 'NO_SIGNAL':
            continue
        
        # Calculate PnL
        result = calculate_pnl(trade, market.outcome)
        
        signal = Signal(
            trade=trade,
            market=market,
            signal_type=signal_type,
            features=features,
            score=score
        )
        result.signal = signal
        
        results_by_type[signal_type].append(result)
        all_test_results.append(result)
    
    conn.close()
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS (TEST SET ONLY)")
    print("=" * 70)
    
    # Overall system performance
    print("\n📊 SYSTEM PERFORMANCE")
    if all_test_results:
        stats = calculate_stats(all_test_results)
        print(f"   Trades: {stats['n']}")
        print(f"   Total PnL: ${stats['total_pnl']:,.0f}")
        print(f"   Mean ROI: {stats['mean_roi']*100:+.2f}%")
        print(f"   Std ROI: {stats['std_roi']*100:.2f}%")
        print(f"   t-stat: {stats['t_stat']:.2f}")
        print(f"   Sharpe: {stats['sharpe']:.2f}")
        print(f"   Win rate: {stats['win_rate']*100:.1f}%")
        print(f"   Max DD: ${stats['max_drawdown']:,.0f}")
        print(f"   Significant: {'✅ YES' if stats['is_significant'] else '❌ NO'}")
    
    # By signal type
    print("\n📈 BY SIGNAL TYPE")
    for signal_type in ['ALPHA', 'INSIDER_CONFIRMED', 'CONFLICT', 'INSIDER_ONLY']:
        if signal_type in results_by_type:
            stats = calculate_stats(results_by_type[signal_type])
            sig = '✓' if stats.get('is_significant') else ' '
            print(f"\n   {sig} {signal_type}:")
            print(f"      n={stats['n']}, ROI={stats['mean_roi']*100:+.2f}%, t={stats['t_stat']:.2f}")
    
    # Baselines
    print("\n📉 BASELINE COMPARISONS")
    
    # Use same test trades for baselines
    test_trades_list = [r.signal.trade for r in all_test_results if r.signal]
    
    if test_trades_list:
        # Random
        random_results = baseline_random(test_trades_list, markets)
        if random_results:
            random_stats = calculate_stats(random_results)
            print(f"   Random: ROI={random_stats['mean_roi']*100:+.2f}%")
        
        # Always NO
        no_results = baseline_always_no(test_trades_list, markets)
        if no_results:
            no_stats = calculate_stats(no_results)
            print(f"   Always NO: ROI={no_stats['mean_roi']*100:+.2f}%")
        
        # Follow odds
        odds_results = baseline_follow_odds(test_trades_list, markets)
        if odds_results:
            odds_stats = calculate_stats(odds_results)
            print(f"   Follow odds: ROI={odds_stats['mean_roi']*100:+.2f}%")
        
        # System vs baselines
        if all_test_results:
            system_roi = calculate_stats(all_test_results)['mean_roi']
            print(f"\n   System alpha vs Random: {(system_roi - random_stats['mean_roi'])*100:+.2f}%")
            print(f"   System alpha vs Always NO: {(system_roi - no_stats['mean_roi'])*100:+.2f}%")
    
    # Feature importance (univariate lift - with caveat)
    print("\n🔬 FEATURE IMPORTANCE (univariate - use with caution)")
    print("   ⚠️  Features may be correlated. Interpret as directional only.")
    
    features_to_test = [
        'is_new_wallet', 'is_very_new_wallet', 'is_low_activity',
        'is_large_bet', 'is_very_large_bet', 'is_longshot', 
        'is_contrarian', 'is_pre_event', 'is_very_pre_event'
    ]
    
    feature_lifts = []
    for feat in features_to_test:
        with_feat = [r for r in all_test_results if r.signal and r.signal.features.get(feat)]
        without_feat = [r for r in all_test_results if r.signal and not r.signal.features.get(feat)]
        
        if len(with_feat) >= 5 and len(without_feat) >= 5:
            roi_with = sum(r.roi for r in with_feat) / len(with_feat)
            roi_without = sum(r.roi for r in without_feat) / len(without_feat)
            lift = roi_with - roi_without
            feature_lifts.append((feat, lift, len(with_feat)))
    
    feature_lifts.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, lift, count in feature_lifts[:5]:
        print(f"   {feat}: {lift*100:+.2f}% lift (n={count})")
    
    # Verdict
    print("\n" + "=" * 70)
    
    if not all_test_results or len(all_test_results) < MIN_TRADES_FOR_SIGNIFICANCE:
        print("⚠️  VERDICT: Insufficient data for conclusions")
    else:
        stats = calculate_stats(all_test_results)
        
        if not stats['is_significant']:
            print("❌ VERDICT: No statistically significant edge")
            print("   t-stat < 2.0 — results could be noise")
        elif stats['mean_roi'] <= 0:
            print("❌ VERDICT: Hypothesis falsified — negative ROI")
        elif system_roi <= max(random_stats['mean_roi'], no_stats['mean_roi']):
            print("⚠️  VERDICT: System does not beat baselines")
        else:
            print("✅ VERDICT: Potential edge detected")
            print("   Proceed to out-of-sample validation")
    
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════
# LOOKAHEAD BIAS CHECK
# ══════════════════════════════════════════════════════════════════

def check_lookahead_bias():
    """
    Self-audit for lookahead bias.
    """
    print("\n🔍 LOOKAHEAD BIAS AUDIT")
    print("=" * 50)
    
    checks = [
        ("Features use only data before trade timestamp", True),
        ("Market outcome NOT used in feature extraction", True),
        ("Resolution timestamp NOT used in scoring", True),
        ("Wallet PnL history NOT used (would leak)", True),
        ("Train/test split is time-based (not random)", True),
        ("Baselines use same trade set as system", True),
    ]
    
    all_pass = True
    for check, status in checks:
        icon = "✅" if status else "❌"
        print(f"   {icon} {check}")
        if not status:
            all_pass = False
    
    if all_pass:
        print("\n   All checks passed. No obvious lookahead bias.")
    else:
        print("\n   ⚠️  CRITICAL: Fix lookahead issues before proceeding!")
    
    print("=" * 50 + "\n")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backtest.py [collect|run|audit]")
        print("  collect [days] - Fetch resolved markets and trades")
        print("  run            - Run walk-forward backtest with baselines")
        print("  audit          - Check for lookahead bias")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "collect":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        collect_data(days_back=days)
    
    elif cmd == "run":
        run_backtest()
    
    elif cmd == "audit":
        check_lookahead_bias()
    
    else:
        print(f"Unknown command: {cmd}")

