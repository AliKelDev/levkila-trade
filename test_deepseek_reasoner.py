import json
import os
from typing import Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

import prompt_builder as pb


def build_prompts() -> Tuple[str, str]:
    """Build system/user prompts without executing trades."""
    state = pb.load_state()
    exchange = pb.build_exchange()

    try:
        current_time = pd.Timestamp.utcnow()
        start_timestamp = state.get("start_timestamp")
        if not start_timestamp:
            start_timestamp = current_time.isoformat()

        minutes_since_start = int(
            max(
                0,
                (current_time - pd.Timestamp(start_timestamp)).total_seconds(),
            )
            // 60
        )
        invocation_count = int(state.get("invocation_count", 0)) + 1

        system_prompt = pb.load_system_prompt()
        market_prompt = pb.build_market_prompt(exchange)
        account_prompt, _, _ = pb.build_account_prompt(exchange, state)
    finally:
        try:
            exchange.close()
        except Exception:  # noqa: BLE001
            pass

    user_prompt = (
        "It has been {minutes} minutes since you started trading. "
        "The current time is {now} and you've been invoked {count} times. "
        "Below, we are providing you with a variety of state data, price data, and "
        "predictive signals so you can discover alpha. Below that is your current "
        "account information, value, performance, positions, etc.\n\n"
    ).format(minutes=minutes_since_start, now=current_time, count=invocation_count)
    user_prompt += "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST\n\n"
    user_prompt += market_prompt
    user_prompt += "\n"
    user_prompt += account_prompt

    return system_prompt, user_prompt


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set")

    system_prompt, user_prompt = build_prompts()

    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 8192,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        pb.DEEPSEEK_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=300,
    )

    print("status:", response.status_code)
    print("headers:", response.headers.get("content-type"))
    print("body:", response.text[:800])

    try:
        data = response.json()
    except ValueError:
        print("Failed to parse JSON body.")
        return

    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content")
    reasoning = message.get("reasoning_content")

    print("\n--- Parsed Content ---")
    print(content)
    print("\n--- Reasoning Content ---")
    print(reasoning)


if __name__ == "__main__":
    main()
