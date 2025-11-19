
# DeepTrade: An LLM-Powered Trading Agent

<p align="center">
  <em>An autonomous crypto-trading agent that leverages the DeepSeek LLM for market analysis and trade execution, monitored through a real-time Dash control panel.</em>
</p>

<p align="center">
  <img width="1440" height="688" alt="DeepTrade Control Panel" src="https://github.com/user-attachments/assets/02dd2ad9-6e25-40fd-9708-7b107706b467" />
</p>

## Core Thesis

This project explores a fundamental question: **Can a Large Language Model, when provided with a rich context of real-time market data and governed by a strict set of rules, function as a viable decision-making engine for short-term algorithmic trading?**

DeepTrade is a framework built to test this hypothesis on the Binance Futures Testnet, focusing on a robust prompt-engineering strategy to guide the LLM's analytical process.

## Architecture Overview

The system operates in a continuous loop, executing the following steps:

1.  **Data Ingestion & Feature Engineering**: The `prompt_builder.py` script connects to the Binance API to fetch the latest OHLCV data for multiple assets (BTC, ETH, etc.) across different timeframes. It then enriches this raw data with a suite of technical indicators using `pandas-ta`.

2.  **Context-Rich Prompt Construction**: All market data, technical indicators, open interest, funding rates, **qualitative news sentiment**, and current account status (positions, P&L) are dynamically assembled into a single, comprehensive prompt. This gives the LLM a full situational snapshot.

3.  **LLM Reasoning & Decision**: The prompt is sent to the `deepseek-reasoner` model. A carefully crafted `system_prompt.md` acts as the model's constitution, defining its rules of engagement, risk parameters, and the mandatory JSON output format. This is where the core trading logic resides.

4.  **Action & Execution**: The bot parses the JSON response from the LLM. Based on the `signal` (e.g., `buy_to_enter`, `close_position`, `hold`), it executes trades via the `ccxt` library, including setting leverage and placing bracket orders (stop-loss and take-profit).

5.  **State Management**: Key information, like the bot's runtime, invocation count, and metadata about open positions (e.g., the exit plan), is persisted in `bot_state.json`. This ensures the agent has memory and context between trading cycles.

6.  **Real-Time Monitoring**: The `dash_app.py` provides a web-based UI to monitor the bot's performance, view the live equity curve, inspect the exact prompts being sent to the LLM, and analyze the model's chain-of-thought reasoning for each decision.

## The Prompting Strategy: The "Brain" of the Bot

The success of this agent hinges entirely on the quality of its prompts. The core of the strategy lies in the `system_prompt.md` file, which constrains the LLM's behavior:

> ```markdown
> PRIMARY DIRECTIVES:
> Follow every rule below exactly as written.
> Rely solely on the numbers provided in the prompt.
> Focus on risk management and precise execution—no extra commentary.
>
> The Exit Plan is Law: When you enter a position, you define an exit_plan containing a profit_target, stop_loss, and an invalidation_condition. Your primary responsibility is to monitor the invalidation_condition. If this condition is met, you MUST issue a close_position action.
> ```

This rigid, rule-based approach transforms the LLM from a generic chatbot into a disciplined trading tool. It is explicitly forbidden from "hallucinating" or acting outside the strict boundaries of its instructions and the provided data.

## Key Features

-   **Multi-Asset, Multi-Timeframe Analysis**: Monitors 6 different cryptocurrencies on both short-term (3-minute) and long-term (4-hour) charts.
-   **Rich Technical Context**: Augments price data with EMA, MACD, RSI, and ATR indicators to inform decisions.
-   **Strict Risk Management**: Every trade entry requires a predefined `exit_plan` with a stop-loss and a clear invalidation condition.
-   **Stateful Operation**: Remembers its own actions and performance across runs for more informed `hold` or `close` decisions.
-   **Interactive Dashboard**: The Dash UI provides a rich, real-time view of the bot's operations, crucial for monitoring and debugging.
-   **Inspectable AI Reasoning**: Leverages the `deepseek-reasoner` model to capture the chain-of-thought, allowing for analysis of *why* the bot made a particular decision.
-   **Qualitative Market Intelligence**: Integrates real-time news and sentiment analysis from Google Gemini to provide the agent with broader market context beyond just price action.

## Getting Started

### Prerequisites
- Python 3.13 (match the version declared in `pyproject.toml`)
- A Binance Futures Testnet account with API credentials
- A DeepSeek API key with access to the `deepseek-chat` model
- (Recommended) [`uv`](https://docs.astral.sh/uv/) for dependency management, or `pip` if you prefer

### Installation
1. Create and activate a virtual environment (recommended):
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   - With `uv`:
     ```bash
     uv sync
     ```
   - With `pip`:
     ```bash
     pip install .
     ```

### Environment Variables
Create a `.env` file in the project root. The script loads it automatically via `python-dotenv`.
```ini
# Binance Futures Testnet
TESTNET_API_KEY=your_binance_testnet_api_key
TESTNET_SECRET_KEY=your_binance_testnet_secret

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key

# Google Gemini (for Sentiment Analysis)
GEMINI_API_KEY=your_gemini_api_key
```

### Running the Bot
To run a single trading cycle from the terminal:
```bash
python prompt_builder.py
```
To launch the interactive dashboard:
```bash
python dash_app.py
```

## Limitations & Future Work

This project serves as a proof-of-concept and has several areas for future exploration:

-   **Backtesting Framework**: The bot currently only paper trades on a testnet. A proper backtesting engine would be needed to validate this strategy's historical performance.
-   **Dynamic Risk Management**: Risk is currently a fixed USD amount per trade. A more advanced model could adjust risk based on market volatility or portfolio size.
-   **Alternative Models**: The framework can be adapted to test other LLMs (e.g., GPT 5.1, Claude Sonnet 4.5) to compare their reasoning and performance.
