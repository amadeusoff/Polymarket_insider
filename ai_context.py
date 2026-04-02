"""
AI Context Layer v2 — Factual context for trade alerts.

Two-step pipeline:
1. Web search → get fresh snippets about the market
2. GPT-4o-mini → summarize into one actionable line

Why two steps:
- GPT alone hallucinates current sports standings, scores, injury reports
- Web search gives real data, GPT just summarizes
- Total cost: ~$0.003 per alert (search free + GPT $0.002)

Only called for alerts passing all filters (~5/day).
"""

import re
import logging
import requests
from typing import Optional

from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# STEP 0: MARKET TYPE DETECTION
# ══════════════════════════════════════════════════════════

SPORTS_KEYWORDS = [
    'nba', 'nfl', 'mlb', 'nhl', 'wnba', 'ncaa', 'epl', 'mls',
    'euroleague', 'ufc', 'tennis', 'golf', ' vs ', ' vs.',
    'champions league', 'la liga', 'serie a', 'bundesliga', 'ligue 1',
    'premier league', 'world cup', 'olympics',
    'hawks', 'celtics', 'lakers', 'warriors', 'nets', 'knicks',
    'bucks', 'bulls', 'magic', 'heat', 'pacers', 'pistons',
    'cavaliers', 'thunder', 'nuggets', 'clippers', 'suns', 'spurs',
    'raptors', 'hornets', 'wizards', 'pelicans', 'kings', 'blazers',
    'grizzlies', 'timberwolves', 'rockets', 'mavericks',
]

POLITICS_KEYWORDS = [
    'president', 'election', 'vote', 'senate', 'congress', 'governor',
    'prime minister', 'parliament', 'referendum', 'party', 'nomination',
    'impeach', 'cabinet', 'minister',
]

CRYPTO_KEYWORDS = [
    'bitcoin', 'ethereum', 'btc', 'eth', 'solana', 'crypto',
    'token', 'defi', 'nft', 'blockchain', 'fdv', 'airdrop',
]

GEOPOLITICS_KEYWORDS = [
    'war', 'strike', 'invasion', 'ceasefire', 'sanctions', 'tariff',
    'iran', 'russia', 'ukraine', 'china', 'taiwan', 'nato', 'military',
]


def detect_market_type(title: str) -> str:
    t = title.lower()
    if any(kw in t for kw in SPORTS_KEYWORDS):
        return "sports"
    if any(kw in t for kw in POLITICS_KEYWORDS):
        return "politics"
    if any(kw in t for kw in CRYPTO_KEYWORDS):
        return "crypto"
    if any(kw in t for kw in GEOPOLITICS_KEYWORDS):
        return "geopolitics"
    return "other"


def extract_search_query(title: str, market_type: str) -> str:
    q = re.sub(r'^Will\s+', '', title, flags=re.IGNORECASE)
    q = re.sub(r'\?$', '', q).strip()

    if market_type == "sports":
        vs_match = re.search(r'(.+?)\s+vs\.?\s+(.+?)(?:\s*[:|\-]|$)', q)
        if vs_match:
            team1 = vs_match.group(1).strip()
            team2 = vs_match.group(2).strip()
            return f"{team1} vs {team2} preview stats 2026"
        return f"{q} preview stats"

    if market_type == "politics":
        return f"{q} polls latest 2026"

    if market_type == "geopolitics":
        return f"{q} latest news 2026"

    return q


# ══════════════════════════════════════════════════════════
# STEP 1: WEB SEARCH (DuckDuckGo HTML, free, no API key)
# ══════════════════════════════════════════════════════════

def web_search_snippets(query: str, max_results: int = 3) -> str:
    """
    Quick web search via DuckDuckGo HTML lite.
    Returns concatenated text snippets.
    Falls back to empty string on any error.
    """
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.debug(f"DDG search returned {resp.status_code}")
            return ""

        # Extract snippets from DDG HTML
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            resp.text,
            re.DOTALL,
        )

        # Clean HTML tags and whitespace
        clean = []
        for s in snippets[:max_results]:
            text = re.sub(r'<[^>]+>', '', s)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 20:
                clean.append(text)

        result = " | ".join(clean)
        return result[:800] if result else ""

    except Exception as e:
        logger.debug(f"Web search failed: {e}")
        return ""


# ══════════════════════════════════════════════════════════
# STEP 2: GPT SUMMARIZATION (type-specific prompts)
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "sports": """You summarize sports context for prediction market traders.
Rules:
- ONE line, max 30 words
- Only facts from the provided search results
- Focus on: standings, recent form (W/L streak), H2H record, key injuries
- If search results don't contain useful data, reply exactly: NO_DATA
- Never invent standings, scores, or stats""",

    "politics": """You summarize political context for prediction market traders.
Rules:
- ONE line, max 30 words
- Only facts from the provided search results
- Focus on: latest polls, key endorsements, recent events
- If search results don't contain useful data, reply exactly: NO_DATA
- Never invent poll numbers or percentages""",

    "geopolitics": """You summarize geopolitical context for prediction market traders.
Rules:
- ONE line, max 30 words
- Only facts from the provided search results
- Focus on: latest diplomatic moves, statements, military developments
- If search results don't contain useful data, reply exactly: NO_DATA
- Never invent quotes or events""",

    "crypto": """You summarize crypto/blockchain context for prediction market traders.
Rules:
- ONE line, max 30 words
- Only facts from the provided search results
- Focus on: price levels, protocol updates, token events, TVL changes
- If search results don't contain useful data, reply exactly: NO_DATA
- Never invent prices or metrics""",

    "other": """You summarize context for prediction market traders.
Rules:
- ONE line, max 30 words
- Only facts from the provided search results
- Focus on the single most relevant fact
- If search results don't contain useful data, reply exactly: NO_DATA
- Never invent facts""",
}


def summarize_with_gpt(
    market_title: str,
    outcome: str,
    odds_pct: float,
    search_results: str,
    market_type: str,
) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None

    system = SYSTEM_PROMPTS.get(market_type, SYSTEM_PROMPTS["other"])

    if search_results:
        user_msg = f"""Market: "{market_title}"
Bet: {outcome} at {odds_pct:.0f}%

Search results:
{search_results}

Summarize the most relevant fact in ONE line:"""
    else:
        user_msg = f"""Market: "{market_title}"
Bet: {outcome} at {odds_pct:.0f}%

No search results available. If you have HIGH confidence factual knowledge about this, give ONE line of context. Otherwise reply: NO_DATA"""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=80,
            temperature=0.2,
        )

        text = response.choices[0].message.content.strip()
        text = text.strip('"').strip("'").strip()

        if not text or "NO_DATA" in text or len(text) < 8:
            return None

        if len(text) > 150:
            text = text[:147] + "..."

        return text

    except Exception as e:
        logger.warning(f"GPT summarization failed: {e}")
        return None


# ══════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def generate_trade_context(
    market_title: str,
    outcome: str,
    odds_pct: float,
    trader_rank: int = 0,
    amount: float = 0,
) -> Optional[str]:
    """
    Generate one-line factual context for a trade alert.
    Pipeline: detect type -> web search -> GPT summarize.
    Returns None on any error (alert sends without context).
    """
    if not market_title:
        return None

    market_type = detect_market_type(market_title)
    logger.info(f"  AI context: type={market_type}, market={market_title[:50]}")

    # Step 1: Web search
    query = extract_search_query(market_title, market_type)
    logger.info(f"  AI context: searching '{query[:60]}'")
    snippets = web_search_snippets(query)

    if snippets:
        logger.info(f"  AI context: got {len(snippets)} chars from web")
    else:
        logger.info(f"  AI context: no search results, GPT-only fallback")

    # Step 2: GPT summarization
    context = summarize_with_gpt(
        market_title=market_title,
        outcome=outcome,
        odds_pct=odds_pct,
        search_results=snippets,
        market_type=market_type,
    )

    if context:
        logger.info(f"  AI context: '{context[:80]}'")

    return context
