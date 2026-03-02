from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SENTINELLE_CACHE_PATH = Path("sentinelle_cache.json")
CYCLE_HISTORY_PATH = Path("cycle_history.json")

SENTINELLE_SYSTEM_PROMPT = """Tu es le Chief Risk Officer - CRO, l'intelligence artificielle d'audit des risques d'un hedge fund quantitatif d'élite. Ton rôle est critique : tu es le filet de sécurité du capital.

Toutes les 5 minutes, tu reçois un rapport sur l'état du marché, les dernières actualités (News), les performances de l'Agent d'Exécution (qui trade en 1-minute), et la Stratégie Macro actuelle (dictée par le CIO pour l'heure en cours).

TA MISSION :
Décider si le contexte actuel invalide la stratégie macroéconomique en cours. Si oui, tu dois déclencher l'alarme pour forcer le CIO à recalculer une nouvelle stratégie en urgence.

RÈGLES D'INTERVENTION (Ne déclenche l'alarme QUE dans ces cas) :
1. Choc Fondamental (News) : Une nouvelle politique, économique ou géopolitique majeure vient de tomber (ex: annonce SEC, faillite, tweet présidentiel influent) qui contredit violemment la stratégie actuelle.
2. Hémorragie Tactique (Drawdown) : L'Agent d'Exécution accumule des pertes anormales (Stop Loss touchés à répétition), prouvant que le biais actuel est déconnecté de la réalité du flux d'ordres.
3. Cygne Noir / Anomalie de Prix (Black Swan) : Le prix ou le volume subit une variation extrême hors des normes statistiques habituelles (Flash Crash ou Pump soudain).

CE QUE TU NE DOIS PAS FAIRE :
- Tu ne dois PAS paniquer face au "bruit" normal du marché (fluctuations mineures de moins de 1%).
- Tu ne passes pas d'ordres de trading toi-même.

FORMAT DE RÉPONSE OBLIGATOIRE :
Tu dois répondre UNIQUEMENT par un objet JSON valide, sans balises markdown (pas de ```json), sans aucun texte avant ou après.
Structure exigée :
{
  "trigger_recalculation": true | false,
  "confidence_in_alarm": [entier de 0 à 100],
  "reasoning_for_logs": "Explication courte et technique de ta décision pour nos logs internes. Si false, explique pourquoi la situation est normale.",
  "message_to_macro_strategist": "Si trigger_recalculation est true, rédige un message d'urgence au CIO expliquant ce qui a changé et ce qu'il doit intégrer dans son recalcul. Si false, mets null."
}"""


def _load_cycle_history() -> List[Dict[str, Any]]:
    """Load the last 10 cycles from disk."""
    if not CYCLE_HISTORY_PATH.exists():
        return []
    try:
        with CYCLE_HISTORY_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data[-10:]
    except Exception:  # noqa: BLE001
        pass
    return []


def _summarise_cycles(cycles: List[Dict[str, Any]]) -> str:
    """Produce a compact, readable summary of the last N cycles for the Sentinelle."""
    if not cycles:
        return "Aucun cycle d'exécution disponible."

    lines = []
    for i, c in enumerate(cycles, 1):
        ts = c.get("run_timestamp", "?")
        inv = c.get("invocation_count", "?")
        bal_before = (c.get("balances_before") or {}).get("total_balance", "?")
        bal_after = (c.get("balances_after") or {}).get("total_balance", "?")
        pnl = (c.get("balances_after") or {}).get("pnl_percent", None)
        decisions = c.get("decisions") or {}

        signals = []
        for coin, payload in decisions.items():
            sig = (payload.get("trade_signal_args") or {}).get("signal", "?")
            signals.append(f"{coin}:{sig}")

        pnl_str = f"{pnl:.2f}%" if pnl is not None else "?"
        lines.append(
            f"[Cycle #{inv} | {ts}] Balance: {bal_before} → {bal_after} | PnL total: {pnl_str} | "
            f"Signaux: {', '.join(signals) if signals else 'aucun'}"
        )
    return "\n".join(lines)


def build_sentinelle_prompt(
    sentiment: str,
    macro_strategy: str,
    market_data: str,
    cycles: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Assemble the user prompt sent to the Sentinelle."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cycle_summary = _summarise_cycles(cycles or [])

    return (
        f"RAPPORT DE SITUATION - AUDIT DES RISQUES (5 MIN)\n"
        f"Heure actuelle : {now}\n\n"
        f"1. STRATÉGIE MACRO EN COURS (À Auditer) :\n"
        f"{macro_strategy.strip() if macro_strategy else 'Aucune stratégie macro disponible.'}\n\n"
        f"2. PERFORMANCE DE L'AGENT D'EXÉCUTION (Depuis le début de la stratégie) :\n"
        f"{cycle_summary}\n\n"
        f"3. DONNÉES DE MARCHÉ ACTUELLES :\n"
        f"{market_data.strip()}\n\n"
        f"4. FLUX D'ACTUALITÉS EN DIRECT (Urgence / Sentiment) :\n"
        f"{sentiment.strip() if sentiment else 'Aucune donnée de sentiment disponible.'}\n\n"
        f"ANALYSE REQUISE :\n"
        f"Sur la base de ces données, la stratégie macro actuelle est-elle toujours sécurisée "
        f"et cohérente avec la réalité du marché ?\n"
        f"Génère immédiatement ta réponse au format JSON strict."
    )


def run_sentinelle(
    market_data: str,
    sentiment: str,
    macro_strategy: str,
) -> Dict[str, Any]:
    """
    Call the Sentinelle (CRO) AI with the latest market context.
    Returns the parsed JSON decision dict, or an error dict on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"trigger_recalculation": False, "reasoning_for_logs": "GEMINI_API_KEY not set."}

    cycles = _load_cycle_history()
    user_prompt = build_sentinelle_prompt(sentiment, macro_strategy, market_data, cycles)

    try:
        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction=SENTINELLE_SYSTEM_PROMPT,
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

        decision = json.loads(raw)

        # Persist the Sentinelle's decision for the dashboard
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "user_prompt": user_prompt,
        }
        temp_path = SENTINELLE_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, ensure_ascii=False)
        temp_path.replace(SENTINELLE_CACHE_PATH)

        print("\n===== SENTINELLE (CRO) =====")
        triggered = decision.get("trigger_recalculation", False)
        confidence = decision.get("confidence_in_alarm", 0)
        reasoning = decision.get("reasoning_for_logs", "")
        print(f"Trigger recalculation: {triggered} | Confidence: {confidence}/100")
        print(f"Reasoning: {reasoning}")

        return decision

    except json.JSONDecodeError as exc:
        print(f"\n===== SENTINELLE ERROR =====\nJSON parse failed: {exc}")
        return {"trigger_recalculation": False, "reasoning_for_logs": f"JSON parse error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        print(f"\n===== SENTINELLE ERROR =====\n{exc}")
        return {"trigger_recalculation": False, "reasoning_for_logs": f"Error: {exc}"}


def get_cached_sentinelle() -> Optional[Dict[str, Any]]:
    """Return the latest Sentinelle decision from cache, or None."""
    if not SENTINELLE_CACHE_PATH.exists():
        return None
    try:
        with SENTINELLE_CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None
