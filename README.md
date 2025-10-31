# DeepTrade

DeepTrade is a small research bot that pulls live market data from the Binance Futures Testnet, packages it as a context-rich prompt, and sends it to DeepSeek for a JSON trading decision. The default configuration monitors BTC, ETH, SOL, BNB, XRP, and DOGE on short (3‑minute) and long (4‑hour) timeframes and augments the raw OHLC data with technical indicators from `pandas-ta`.

## Prerequisites
- Python 3.13 (match the version declared in `pyproject.toml`)
- A Binance Futures Testnet account with API credentials
- A DeepSeek API key with access to the `deepseek-chat` model
- (Recommended) [`uv`](https://docs.astral.sh/uv/) for dependency management, or `pip` if you prefer

## Installation
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

## Environment Variables
Create a `.env` file in the project root. The script loads it automatically via `python-dotenv`.
```ini
# Binance Futures Testnet
TESTNET_API_KEY=your_binance_testnet_api_key
TESTNET_SECRET_KEY=your_binance_testnet_secret

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### Getting Binance Testnet Credentials
1. Go to the [Binance Futures Testnet](https://testnet.binancefuture.com/) and register/log in.
2. Navigate to **API Key** management and create a new key pair for demo trading.
3. Enable Futures trading permissions for the key and note the generated `API Key` and `Secret Key`.
4. Optionally top up your demo USDT balance via the Testnet “Futures” wallet faucet.
5. Store the credentials securely (the `.gitignore` already ignores `.env` and `*.pem` files).

If you already manage credentials as `.pem` files, place them in this directory; the defaults (`binance_testnet_private.pem`, `binance_testnet_public.pem`) are ignored by Git.

### Getting a DeepSeek API Key
1. Visit the [DeepSeek dashboard](https://platform.deepseek.com/) and create an account.
2. Generate an API key with permission to call the `deepseek-chat` model.
3. Copy the key into your `.env` under `DEEPSEEK_API_KEY`.

## Running the Bot
```bash
python prompt_builder.py
```
What you’ll see in the terminal:
1. The full user prompt that combines market data and your current account state.
2. The raw DeepSeek response (including any `<chain_of_thought>` block) and the parsed JSON decisions.
3. Order execution logs covering leverage updates, entries, exits, and bracket orders.
4. A refreshed account summary after trades settle.

Under the hood each invocation will:
- Connect to the Binance Futures Testnet with demo trading enabled.
- Fetch OHLC data, compute indicators, and gather balances/positions.
- Persist local metadata (confidence, exit plan, order ids) in `bot_state.json` so the next turn can echo the same plan back to the LLM.
- Track the bot start time and invocation count in `bot_state.json` so prompts include runtime context.
- Respect the LLM’s `trade_signal_args` by sizing positions from `risk_usd`, placing market entries, and attaching reduce-only stop-loss and take-profit orders.

The script currently runs a single evaluation/decision cycle. Wrap `main()` in your own scheduler if you need a continuous loop.

## Customisation
- Adjust `COINS`, `INTRADAY_TIMEFRAME`, indicator settings, or the prompt copy directly in `prompt_builder.py`.
- Change risk sizing by editing `calculate_order_amount` (e.g. incorporate margin requirements or volatility filters).
- Swap bracket behaviour in `place_bracket_orders` if you prefer limit-based targets or OCO logic.
- Extend `bot_state.json` with additional analytics you want the LLM to remember between turns.

## Safety Notes
- Never commit `.env`, `.pem`, or `bot_state.json` files; they may contain secrets or live order ids.
- Demo keys are separate from production – keep them isolated and rotate regularly.
- Binance Testnet still enforces min notional and leverage limits. Watch the terminal output for rejection warnings.
- Always verify orders on the Testnet UI before trusting a new strategy variant in production.
