"""
AI Context Layer v3 — Factual context for trade alerts.

GPT-4o-mini only (no web search — DDG blocked in GitHub Actions).
Asks for GENERAL KNOWLEDGE context, not real-time stats.

What GPT can reliably provide:
- Sports: team tier (contender vs rebuilding), conference, general strength
- Politics: candidate background, party dynamics
- Crypto: project description, market cap tier
- Geopolitics: situation background, key actors

What it CANNOT provide (and shouldn't try):
- Today's injury report, exact standings, live scores
- Current poll numbers, exact vote counts
- Real-time prices

Cost: ~$0.002 per call. Only for alerts passing all filters.
"""

import re
import logging
from typing import Optional

from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# MARKET TYPE DETECTION
# ══════════════════════════════════════════════════════════

def detect_market_type(title: str) -> str:
    t = title.lower()
    sports = [
        'nba', 'nfl', 'mlb', 'nhl', 'wnba', 'ncaa', 'epl', 'mls',
        'euroleague', 'ufc', 'tennis', 'golf', ' vs ', ' vs.',
        'champions league', 'la liga', 'serie a', 'bundesliga',
        'premier league', 'world cup', 'cricket',
    ]
    politics = [
        'president', 'election', 'vote', 'senate', 'congress',
        'governor', 'prime minister', 'parliament', 'party',
        'nomination', 'impeach', 'minister',
    ]
    crypto = [
        'bitcoin', 'ethereum', 'btc', 'eth', 'solana', 'crypto',
        'token', 'defi', 'nft', 'fdv', 'airdrop',
    ]
    geo = [
        'war', 'strike', 'invasion', 'ceasefire', 'sanctions',
        'tariff', 'iran', 'russia', 'ukraine', 'china', 'taiwan',
        'nato', 'military',
    ]
    if any(kw in t for kw in sports):
        return "sports"
    if any(kw in t for kw in politics):
        return "politics"
    if any(kw in t for kw in crypto):
        return "crypto"
    if any(kw in t for kw in geo):
        return "geopolitics"
    return "other"


# ══════════════════════════════════════════════════════════
# TYPE-SPECIFIC PROMPTS
# ══════════════════════════════════════════════════════════

PROMPTS = {
    "sports": """You are an independent sports analyst. Your job is to CHALLENGE bets, not confirm them.

Market: "{title}"
Bet: {outcome} at {odds:.0f}%

Is {odds:.0f}% FAIR, TOO HIGH, or TOO LOW for {outcome}?
Consider: team tier, home/away patterns, conference strength, historical matchups.
Start your answer with one of: FAIR | OVERPRICED | UNDERPRICED
Then explain WHY in max 20 words. Be specific.
Do NOT invent W-L records or injury reports. If you don't know these teams: NO_DATA""",

    "politics": """You are an independent political analyst. Your job is to CHALLENGE bets, not confirm them.

Market: "{title}"
Bet: {outcome} at {odds:.0f}%

Is {odds:.0f}% FAIR, TOO HIGH, or TOO LOW for {outcome}?
Consider: incumbent advantage, party dynamics, historical precedent, candidate strength.
Start your answer with one of: FAIR | OVERPRICED | UNDERPRICED
Then explain WHY in max 20 words. Do NOT invent poll numbers. If unsure: NO_DATA""",

    "crypto": """You are an independent crypto analyst. Your job is to CHALLENGE bets, not confirm them.

Market: "{title}"
Bet: {outcome} at {odds:.0f}%

Is {odds:.0f}% FAIR, TOO HIGH, or TOO LOW for {outcome}?
Consider: project fundamentals, market cap tier, narrative momentum, volatility.
Start your answer with one of: FAIR | OVERPRICED | UNDERPRICED
Then explain WHY in max 20 words. Do NOT invent prices. If unsure: NO_DATA""",

    "geopolitics": """You are an independent geopolitical analyst. Your job is to CHALLENGE bets, not confirm them.

Market: "{title}"
Bet: {outcome} at {odds:.0f}%

Is {odds:.0f}% FAIR, TOO HIGH, or TOO LOW for {outcome}?
Consider: diplomatic trajectory, military posture, historical precedent, escalation risk.
Start your answer with one of: FAIR | OVERPRICED | UNDERPRICED
Then explain WHY in max 20 words. Do NOT invent events. If unsure: NO_DATA""",

    "other": """You are an independent analyst. Your job is to CHALLENGE bets, not confirm them.

Market: "{title}"
Bet: {outcome} at {odds:.0f}%

Is {odds:.0f}% FAIR, TOO HIGH, or TOO LOW for {outcome}?
Start your answer with one of: FAIR | OVERPRICED | UNDERPRICED
Then explain WHY in max 20 words. If unsure: NO_DATA""",
}


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def generate_trade_context(
    market_title: str,
    outcome: str,
    odds_pct: float,
    trader_rank: int = 0,
    amount: float = 0,
) -> Optional[str]:
    """
    Generate one-line context for a trade alert.
    Returns None on error or if GPT has no useful context.
    """
    if not market_title or not OPENAI_API_KEY:
        return None

    market_type = detect_market_type(market_title)

    prompt_template = PROMPTS.get(market_type, PROMPTS["other"])
    prompt = prompt_template.format(
        title=market_title,
        outcome=outcome,
        odds=odds_pct,
    )

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()
        text = text.strip('"').strip("'").strip()

        if not text or "NO_DATA" in text or len(text) < 8:
            logger.info(f"  AI context: NO_DATA for '{market_title[:50]}'")
            return None

        if len(text) > 150:
            text = text[:147] + "..."

        logger.info(f"  AI context [{market_type}]: {text[:80]}")
        return text

    except Exception as e:
        logger.warning(f"AI context failed: {e}")
        return None
