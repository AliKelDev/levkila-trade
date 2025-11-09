import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

import prompt_builder as pb


def build_prompts() -> Tuple[str, str, pd.Timestamp]:
    """Construct system and user prompts without invoking the LLM."""
    state = pb.load_state()
    exchange = pb.build_exchange()

    try:
        current_time = pd.Timestamp.utcnow()
        start_timestamp = state.get("start_timestamp")
        if start_timestamp:
            start = pd.Timestamp(start_timestamp)
        else:
            start = current_time

        minutes_since_start = int(
            max(0, (current_time - start).total_seconds()) // 60
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

    return system_prompt, user_prompt, current_time


def write_markdown(system_prompt: str, user_prompt: str, output_path: Path, timestamp: pd.Timestamp) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "# Prompt Snapshot\n"
    contents += f"Generated at {timestamp.isoformat()} UTC\n\n"
    contents += "## System Prompt\n\n"
    contents += "```text\n"
    contents += system_prompt
    if not system_prompt.endswith("\n"):
        contents += "\n"
    contents += "```\n\n"
    contents += "## User Prompt\n\n"
    contents += "```text\n"
    contents += user_prompt
    if not user_prompt.endswith("\n"):
        contents += "\n"
    contents += "```\n"
    output_path.write_text(contents, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompts and export them as Markdown")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/prompt_snapshot.md"),
        help="Path to the markdown file that will receive the prompt contents.",
    )
    args = parser.parse_args()

    load_dotenv()
    system_prompt, user_prompt, timestamp = build_prompts()
    write_markdown(system_prompt, user_prompt, args.output, timestamp)
    print(f"Prompt snapshot written to {args.output}")


if __name__ == "__main__":
    main()
