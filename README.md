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
The script will:
1. Connect to Binance Futures Testnet using the credentials from `.env`.
2. Fetch the configured symbols and compute indicators (EMA, MACD, RSI, ATR).
3. Gather account balances and open positions.
4. Assemble a structured prompt (market data + account state) for the LLM.
5. Load `system_prompt.md` and send both the system and user prompts to DeepSeek.
6. Print the JSON decision returned by DeepSeek.

The loop currently runs once per invocation. You can adapt the commented main loop for continuous operation.

## Customisation
- Edit `COINS`, `INTRADAY_TIMEFRAME`, or indicator lengths in `prompt_builder.py` to target different markets or speeds.
- Update `system_prompt.md` for alternate trading policies or evaluation criteria.
- Extend the script with real order execution logic if you want to progress beyond prompt experimentation.

## Safety Notes
- Never commit `.env` or credential files. The included `.gitignore` enforces this.
- Keep your Testnet and DeepSeek keys separate from production keys.
- Always test strategy changes on the Testnet before going live.
