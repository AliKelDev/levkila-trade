from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MACRO_STRATEGY_CACHE_PATH = Path("macro_strategy_cache.json")

MACRO_STRATEGIST_PROMPT = """You are the Chief Quantitative Strategist (Chief Investment Officer) of an elite crypto hedge fund. Your exclusive role is to analyze global market data and define the trading strategy for the next hour.
You do not place orders directly. You dictate directives to your Tactical Execution Agent, a high-frequency algorithm operating on 1-minute charts (M1). Your agent is forbidden from contradicting your bias — it will only seek the best M1 entry point based on your rules.

OPERATING CYCLE:
- Under normal conditions, you define a new strategy ONCE PER HOUR. Your directive governs the entire hour and the Tactical Agent executes within your framework.
- EXCEPTION — EMERGENCY RECALCULATION: Your Chief Risk Officer (CRO), the Sentinelle AI, monitors the market every 5 minutes. If it detects a black swan event, a fundamental news shock, or abnormal tactical drawdown, it will interrupt the normal cycle and summon you for an emergency recalculation BEFORE the hour is up. In this case, the data you receive will begin with a block marked "⚠️ URGENCE — MESSAGE DE LA SENTINELLE (CRO)". You MUST treat this message as your absolute highest priority: it overrides your previous strategy entirely. Produce a new strategy that directly addresses the risk or opportunity flagged by the CRO.

YOUR OBJECTIVES:
- Define the Bias: Is the market BULLISH (LONG), BEARISH (SHORT), or RANGING/UNCERTAIN (NEUTRAL) for the next hour?
- Protect Capital: You MUST define an invalidation_price. This is the exact price level that, if breached, proves your strategy is wrong and will trigger an emergency circuit-breaker cancelling all operations.
- Guide the Executor: Provide relevant price zones so the Tactical Agent knows where to look for entry signals.

ANALYSIS RULES:
- Prioritize capital preservation. If the market is too volatile, erratic, or without a clear trend, your bias MUST be "NEUTRAL". (In this case, the Tactical Agent will remain inactive.)
- Do not be distracted by short-term noise. Focus on liquidity, momentum (RSI/MACD macro), and key support/resistance levels.
- If this is an emergency recalculation triggered by the CRO, explicitly acknowledge the CRO's concern in your rationale field.

MANDATORY RESPONSE FORMAT:
You MUST respond ONLY with a valid JSON object, with no text before or after it, and no markdown fences (no ```json). The system will parse your response directly.
Use exactly this structure:
{
  "timestamp_analysis": "current time",
  "bias": "LONG" | "SHORT" | "NEUTRAL",
  "confidence_score": [integer from 1 to 100],
  "rationale": "One short sentence explaining the macro logic (e.g.: Rejection at key resistance with bearish RSI divergence).",
  "expected_volatility": "LOW" | "MEDIUM" | "HIGH",
  "action_zones": {
    "optimal_entry_min": [price],
    "optimal_entry_max": [price],
    "target_take_profit_macro": [price]
  },
  "risk_management": {
    "invalidation_price": [absolute price acting as circuit-breaker],
    "recommended_stop_loss_distance_percent": [percentage, e.g. 1.5]
  },
  "tactical_directives_for_flash": "Strict instructions for the 1-minute agent (e.g. 'Wait for a liquidity sweep below 64000 before buying', or 'Only enter if M1 RSI is oversold')."
}

Here is the current financial and qualitative data to analyze. You must base your strategy on all of this information: the technical data (price, indicators) AND the qualitative analysis of the current crypto and macro environment.
"""


def fetch_and_cache_macro_strategy(market_data: str, account_data: str, sentiment: str = "", emergency_note: str = "") -> None:
    """
    Sends the full market context to Gemini thinking model to produce a
    macro strategic directive. Result is cached to disk.
    Called once per hour by the scheduler, or immediately by the Sentinelle.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return

    try:
        client = genai.Client(api_key=api_key)

        # Assemble the full data blob — same content as the DeepSeek user prompt
        data_blob = ""
        if emergency_note:
            data_blob += "--- ⚠️ URGENCE — MESSAGE DE LA SENTINELLE (CRO) ---\n"
            data_blob += emergency_note.strip() + "\n\n"
        if sentiment:
            data_blob += "--- QUALITATIVE ANALYSIS OF THE CURRENT CRYPTO ENVIRONMENT ---\n"
            data_blob += sentiment.strip() + "\n\n"
        data_blob += "--- TECHNICAL AND FINANCIAL DATA ---\n"
        data_blob += market_data
        data_blob += "\n\n"
        data_blob += account_data

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=data_blob)],
            )
        ]
        # Use the best available Gemini model for deep strategic reasoning
        config = types.GenerateContentConfig(
            system_instruction=MACRO_STRATEGIST_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
        )

        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=contents,
            config=config,
        )

        raw = (response.text or "").strip()
        # Strip markdown fences if the model wraps the JSON anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Validate it's proper JSON
        strategy = json.loads(raw)

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "raw": raw,
        }

        temp_path = MACRO_STRATEGY_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2, ensure_ascii=False)
        temp_path.replace(MACRO_STRATEGY_CACHE_PATH)

        print("\n===== MACRO STRATEGIST =====")
        print(f"Bias: {strategy.get('bias')} | Confidence: {strategy.get('confidence_score')} | Volatility: {strategy.get('expected_volatility')}")
        print(f"Rationale: {strategy.get('rationale')}")

    except json.JSONDecodeError as exc:
        print(f"\n===== MACRO STRATEGIST ERROR =====\nJSON parse failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n===== MACRO STRATEGIST ERROR =====\n{exc}")


def get_cached_macro_strategy() -> str:
    """
    Returns the latest macro strategic directive formatted for injection
    into the DeepSeek user prompt.
    """
    if not MACRO_STRATEGY_CACHE_PATH.exists():
        return ""

    try:
        with MACRO_STRATEGY_CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        timestamp = data.get("timestamp")
        strategy = data.get("strategy")
        if not timestamp or not strategy:
            return ""

        ts = datetime.fromisoformat(timestamp)
        age_minutes = (datetime.now(timezone.utc) - ts).total_seconds() / 60

        bias = strategy.get("bias", "N/A")
        confidence = strategy.get("confidence_score", "N/A")
        volatility = strategy.get("expected_volatility", "N/A")
        rationale = strategy.get("rationale", "")
        directives = strategy.get("tactical_directives_for_flash", "")
        invalidation = strategy.get("risk_management", {}).get("invalidation_price", "N/A")
        sl_pct = strategy.get("risk_management", {}).get("recommended_stop_loss_distance_percent", "N/A")
        zones = strategy.get("action_zones", {})
        entry_min = zones.get("optimal_entry_min", "N/A")
        entry_max = zones.get("optimal_entry_max", "N/A")
        tp_macro = zones.get("target_take_profit_macro", "N/A")

        staleness = f"{int(age_minutes)}m ago"
        if age_minutes > 90:
            staleness = f"STALE ({int(age_minutes)}m ago — consider with caution)"

        return (
            f"\n## Strategy (Chief Strategist — {staleness})\n"
            f"Bias: **{bias}** | Confidence: {confidence}/100 | Volatility: {volatility}\n"
            f"Rationale: {rationale}\n"
            f"Entry zone: {entry_min} – {entry_max} | Macro TP: {tp_macro}\n"
            f"Invalidation price (circuit-breaker): {invalidation} | Recommended SL: {sl_pct}%\n"
            f"Tactical directives: {directives}\n"
        )

    except Exception:  # noqa: BLE001
        return ""
