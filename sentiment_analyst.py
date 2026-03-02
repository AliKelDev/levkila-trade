from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SENTIMENT_CACHE_PATH = Path("sentiment_snapshot.json")
COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]

# Free RSS feeds from top crypto media — no API key required
RSS_FEEDS = [
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("CoinDesk",      "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt",       "https://decrypt.co/feed"),
    ("The Block",     "https://www.theblock.co/rss.xml"),
]

_RELEVANT_KEYWORDS = {
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "bnb", "xrp", "doge",
    "crypto", "blockchain", "defi", "nft", "altcoin", "binance", "coinbase",
    "sec", "fed", "inflation", "cpi", "interest rate", "etf", "stablecoin",
    "liquidat", "whale", "open interest", "funding rate", "flash crash",
    "regulation", "hack", "exploit",
}

RSS_MAX_AGE_HOURS = 3
RSS_MAX_ARTICLES = 15

SENTIMENT_SYSTEM_PROMPT = """You are the Lead Market Intelligence Analyst for DeepTrade, an elite algorithmic crypto hedge fund.
Your sole objective is to extract high-fidelity, actionable trading sentiment from real-time news, filtering out market noise, retail hype, and unverified rumors.

YOUR INPUTS:
1. RSS Feeds (Primary Source): You will receive a stream of the latest RSS headlines and snippets in the user prompt.
2. Web Search Tool (Verification & Context): You have access to the web. You MUST use it to verify shocking/urgent RSS headlines, fetch missing macro data (e.g., current DXY, S&P500 futures, latest Fed/CPI prints), or investigate specific asset catalysts.

YOUR ANALYTICAL FRAMEWORK (Signal vs. Noise):
- Ignore generic crypto-influencer opinions, clickbait, and low-tier blogs.
- Focus strictly on INSTITUTIONAL FLOW catalysts: Regulatory actions (SEC, CFTC), Macro-economics (Interest rates, inflation data, geopolitical shocks), Capital flows (ETF inflows/outflows, whale movements, exchange liquidations), and Core protocol upgrades.
- Evaluate the Impact Duration: Is this a 5-minute algorithmic reaction (e.g., a sudden tweet) or a multi-day structural shift (e.g., a BlackRock filing)?
- Contradiction Check: If RSS feeds show panic but macro data (S&P500/DXY) is stable, highlight this divergence.

YOUR BEHAVIORAL RULES:
1. Fact-Check Everything: If an RSS headline claims a "Black Swan" event (e.g., "Exchange X hacked", "SEC bans Y"), trigger a web search immediately to cross-reference multiple top-tier financial sources (Bloomberg, Reuters, WSJ, CoinDesk) before flagging it as true.
2. Extreme Objectivity: Do not use emotional language. Use quantitative, cold, and precise financial terminology.
3. Default to Neutral: If the news is mundane or sideways, confidently declare a NEUTRAL bias. Do not force a bullish or bearish narrative if the data does not support it.

YOUR OUTPUT FORMAT:
You must respond EXCLUSIVELY in valid JSON format, adhering strictly to the schema below. No markdown, no conversational text before or after the JSON.

{
  "timestamp_analysis": "YYYY-MM-DDTHH:MM:SSZ",
  "global_macro_environment": {
    "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
    "dxy_sp500_context": "Brief summary of traditional finance impact on crypto today."
  },
  "crypto_market_sentiment": {
    "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
    "confidence_score": 0-100,
    "dominant_narrative": "One sentence describing the current market driver."
  },
  "verified_catalysts": [
    {
      "headline": "Exact or summarized headline",
      "source_type": "RSS" | "Web Search",
      "impact_level": "HIGH" | "MEDIUM" | "LOW",
      "affected_assets": ["BTC", "ETH", "ALL"],
      "catalyst_nature": "MACRO" | "REGULATORY" | "ON-CHAIN" | "TECHNICAL",
      "description": "Concise explanation of why this moves the market."
    }
  ],
  "asset_specific_sentiment": {
    "BTC": {"bias": "BULLISH" | "BEARISH" | "NEUTRAL", "rationale": "..."},
    "ETH": {"bias": "BULLISH" | "BEARISH" | "NEUTRAL", "rationale": "..."},
    "SOL": {"bias": "BULLISH" | "BEARISH" | "NEUTRAL", "rationale": "..."},
    "BNB": {"bias": "BULLISH" | "BEARISH" | "NEUTRAL", "rationale": "..."},
    "XRP": {"bias": "BULLISH" | "BEARISH" | "NEUTRAL", "rationale": "..."},
    "DOGE": {"bias": "BULLISH" | "BEARISH" | "NEUTRAL", "rationale": "..."}
  },
  "black_swan_alert": {
    "triggered": true | false,
    "urgency_message": "Null if false. If true, write a direct, urgent warning for the Chief Risk Officer (CRO)."
  }
}"""


# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------

def _symbol_to_coin(symbol: str) -> str:
    return symbol.split("/")[0]


def fetch_rss_headlines(
    max_age_hours: int = RSS_MAX_AGE_HOURS,
    max_articles: int = RSS_MAX_ARTICLES,
) -> str:
    """Fetch recent crypto headlines from free RSS feeds and return a formatted block."""
    import calendar

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    articles = []

    for source_name, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub_date: Optional[datetime] = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime.fromtimestamp(
                        calendar.timegm(entry.published_parsed), tz=timezone.utc
                    )
                if pub_date and pub_date < cutoff:
                    continue

                title = (getattr(entry, "title", "") or "").strip()
                summary = (getattr(entry, "summary", "") or "").strip()
                link = (getattr(entry, "link", "") or "").strip()

                combined = (title + " " + summary).lower()
                if not any(kw in combined for kw in _RELEVANT_KEYWORDS):
                    continue

                articles.append({
                    "source": source_name,
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                })
        except Exception:  # noqa: BLE001
            continue

    if not articles:
        return ""

    articles.sort(
        key=lambda a: a["pub_date"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    articles = articles[:max_articles]

    lines = ["=== LIVE NEWS FEED (specialist crypto media, last 3h) ==="]
    for art in articles:
        age_str = ""
        if art["pub_date"]:
            age_min = int((datetime.now(timezone.utc) - art["pub_date"]).total_seconds() / 60)
            age_str = f" [{age_min}m ago]"
        lines.append(f"[{art['source']}]{age_str} {art['title']}")
        if art["link"]:
            lines.append(f"  → {art['link']}")
    lines.append("=== END LIVE NEWS FEED ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch_and_cache_sentiment(coins: List[str] = COINS) -> None:
    """
    Background job: fetches RSS headlines, then calls Gemini (with Google Search grounding)
    to produce a structured JSON sentiment analysis. Result is cached to disk.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return

    try:
        client = genai.Client(api_key=api_key)
        coin_names = ", ".join(_symbol_to_coin(symbol) for symbol in coins)

        # 1. Fetch RSS headlines (fast, no API cost)
        rss_block = fetch_rss_headlines()

        # 2. Build user prompt
        prompt_parts: list[str] = []
        if rss_block:
            prompt_parts.append(
                "The following headlines were just published by specialist crypto media. "
                "Use them as factual anchors. Fact-check any alarming ones via web search.\n\n"
                + rss_block
                + "\n"
            )
        prompt_parts.append(
            f"Coins monitored by our fund: {coin_names}.\n"
            "Now perform your full analysis using the web search tool to retrieve "
            "any missing data (DXY, S&P500 futures, latest macro prints, ETF flows). "
            "Respond ONLY with the required JSON object."
        )
        user_prompt = "\n".join(prompt_parts)

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction=SENTIMENT_SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=contents,
            config=config,
        )
        raw = (response.text or "").strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Try to parse as JSON; fall back to storing raw text
        sentiment_json: Optional[Dict[str, Any]] = None
        try:
            sentiment_json = json.loads(raw)
        except json.JSONDecodeError:
            pass

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_json": sentiment_json,   # structured (None if parse failed)
            "content_raw": raw,               # always stored for debugging
            "rss_headlines_injected": bool(rss_block),
            "status": "fresh",
        }

        temp_path = SENTIMENT_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, ensure_ascii=False)
        temp_path.replace(SENTIMENT_CACHE_PATH)

        # Log summary
        print("\n===== SENTIMENT ANALYST =====")
        if sentiment_json:
            macro_bias = sentiment_json.get("global_macro_environment", {}).get("bias", "?")
            crypto_bias = sentiment_json.get("crypto_market_sentiment", {}).get("bias", "?")
            confidence = sentiment_json.get("crypto_market_sentiment", {}).get("confidence_score", "?")
            black_swan = sentiment_json.get("black_swan_alert", {}).get("triggered", False)
            print(f"Macro: {macro_bias} | Crypto: {crypto_bias} | Confidence: {confidence}/100 | Black Swan: {black_swan}")
        else:
            print("Warning: Gemini did not return valid JSON. Raw text stored.")

    except Exception as exc:  # noqa: BLE001
        print(f"\n===== SENTIMENT ERROR =====\n{exc}")


# ---------------------------------------------------------------------------
# Cache readers
# ---------------------------------------------------------------------------

def get_cached_sentiment_json() -> Optional[Dict[str, Any]]:
    """Return the structured JSON dict from the last sentiment analysis, or None."""
    if not SENTIMENT_CACHE_PATH.exists():
        return None
    try:
        with SENTIMENT_CACHE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("content_json")
    except Exception:  # noqa: BLE001
        return None


def get_cached_sentiment() -> str:
    """
    Returns formatted qualitative intelligence for DeepSeek / Macro Strategist.
    Converts the structured JSON into a readable, comprehensive text block.
    """
    if not SENTIMENT_CACHE_PATH.exists():
        return ""
    try:
        with SENTIMENT_CACHE_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        timestamp = data.get("timestamp")
        content_json: Optional[Dict[str, Any]] = data.get("content_json")
        content_raw: str = data.get("content_raw", "")

        if not timestamp:
            return ""

        ts = datetime.fromisoformat(timestamp)
        age_minutes = int((datetime.now(timezone.utc) - ts).total_seconds() / 60)
        label = "Latest Intelligence" if age_minutes <= 60 else "Old Intelligence (Context Only)"

        header = f"\n### QUALITATIVE INTELLIGENCE ({label} — {age_minutes}m ago)\n"

        # If we have structured JSON, format it nicely
        if content_json:
            lines = [header]

            macro = content_json.get("global_macro_environment", {})
            crypto = content_json.get("crypto_market_sentiment", {})
            lines.append(f"**MACRO ENVIRONMENT:** {macro.get('bias', '?')} — {macro.get('dxy_sp500_context', '')}")
            lines.append(
                f"**CRYPTO SENTIMENT:** {crypto.get('bias', '?')} "
                f"(confidence: {crypto.get('confidence_score', '?')}/100) — "
                f"{crypto.get('dominant_narrative', '')}"
            )

            catalysts = content_json.get("verified_catalysts", [])
            if catalysts:
                lines.append("\n**VERIFIED CATALYSTS:**")
                for c in catalysts:
                    assets = ", ".join(c.get("affected_assets", []))
                    lines.append(
                        f"  [{c.get('impact_level','?')}] [{c.get('catalyst_nature','?')}] "
                        f"{c.get('headline','?')} (assets: {assets})"
                    )
                    if c.get("description"):
                        lines.append(f"    → {c['description']}")

            asset_sent = content_json.get("asset_specific_sentiment", {})
            if asset_sent:
                lines.append("\n**ASSET-SPECIFIC SENTIMENT:**")
                for coin, info in asset_sent.items():
                    lines.append(f"  {coin}: {info.get('bias','?')} — {info.get('rationale','')}")

            black_swan = content_json.get("black_swan_alert", {})
            if black_swan.get("triggered"):
                lines.append(f"\n⚠️ **BLACK SWAN ALERT:** {black_swan.get('urgency_message', '')}")

            return "\n".join(lines) + "\n"

        # Fallback: return raw text
        if content_raw:
            return header + content_raw + "\n"

        return ""

    except Exception:  # noqa: BLE001
        return ""
