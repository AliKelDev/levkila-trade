import ccxt
import pandas as pd
import pandas_ta as ta
import os
import time
from dotenv import load_dotenv
import json
import requests

# --- Load Environment Variables ---
load_dotenv() 

# --- Configuration ---
EXCHANGE = 'binance'
COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'DOGE/USDT']
INTRADAY_TIMEFRAME = '3m'
LONGTERM_TIMEFRAME = '4h'
INTRADAY_BARS = 10
LONGTERM_BARS = 10
SYSTEM_PROMPT_FILENAME = "system_prompt.md"

# Load credentials from .env file for the Mock Trading environment
API_KEY = os.getenv("TESTNET_API_KEY") 
SECRET_KEY = os.getenv("TESTNET_SECRET_KEY")

# --- Authenticated Exchange Connection (The OFFICIAL CCXT Way) ---
print("Connecting to Binance Mock Trading...")
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'options': {
        'defaultType': 'future',
    },
})

# THIS IS THE NEW, CORRECT, OFFICIAL METHOD
exchange.enable_demo_trading(True) 

print("Connection successful.")

# --- Load System Prompt from File ---
# We do this ONCE at startup.
print(f"Loading system prompt from '{SYSTEM_PROMPT_FILENAME}'...")
try:
    with open(SYSTEM_PROMPT_FILENAME, 'r') as f:
        SYSTEM_PROMPT = f.read()
    print("System prompt loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: System prompt file '{SYSTEM_PROMPT_FILENAME}' not found.")
    exit() # Crash the bot if the prompt is missing. This is a critical error.


# --- Bot Functions ---

def get_market_data(exchange_instance):
    """
    Fetches and formats market data.
    """
    master_prompt = "CURRENT MARKET STATE FOR ALL COINS\n"
    master_prompt += "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST\n\n"

    for coin in COINS:
        symbol = coin.split('/')[0]
        try:
            intraday_ohlcv = exchange_instance.fetch_ohlcv(coin, timeframe=INTRADAY_TIMEFRAME, limit=100)
            longterm_ohlcv = exchange_instance.fetch_ohlcv(coin, timeframe=LONGTERM_TIMEFRAME, limit=100)

            if not intraday_ohlcv or not longterm_ohlcv:
                print(f"Could not fetch data for {coin}. Skipping.")
                continue

            intraday_df = pd.DataFrame(intraday_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            longterm_df = pd.DataFrame(longterm_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # ROBUST INDICATOR CODE
            for df in [intraday_df, longterm_df]:
                df.ta.ema(length=20, append=True)
                df.ta.ema(length=50, append=True)
                df.ta.macd(fast=12, slow=26, signal=9, append=True)
                df.ta.rsi(length=7, append=True)
                df.ta.rsi(length=14, append=True)
                df.ta.atr(length=3, append=True)
                df.ta.atr(length=14, append=True)

            latest_intraday = intraday_df.iloc[-1]
            latest_longterm = longterm_df.iloc[-1]
            intraday_series = intraday_df.tail(INTRADAY_BARS)
            longterm_series = longterm_df.tail(LONGTERM_BARS)

            coin_prompt = f"ALL {symbol} DATA\n"
            coin_prompt += f"current_price = {latest_intraday['close']}, current_ema20 = {latest_intraday['EMA_20']:.3f}, current_macd = {latest_intraday['MACDh_12_26_9']:.3f}, current_rsi (7 period) = {latest_intraday['RSI_7']:.3f}\n\n"
            coin_prompt += "Intraday series (by minute, oldest → latest):\n"
            coin_prompt += f"Mid prices: {intraday_series['close'].tolist()}\n"
            coin_prompt += f"EMA indicators (20‑period): {[round(x, 3) for x in intraday_series['EMA_20'].tolist()]}\n"
            coin_prompt += f"MACD indicators: {[round(x, 3) for x in intraday_series['MACDh_12_26_9'].tolist()]}\n"
            coin_prompt += f"RSI indicators (7‑Period): {[round(x, 3) for x in intraday_series['RSI_7'].tolist()]}\n\n"
            coin_prompt += "Longer‑term context (4‑hour timeframe):\n"
            coin_prompt += f"20‑Period EMA: {latest_longterm['EMA_20']:.3f} vs. 50‑Period EMA: {latest_longterm['EMA_50']:.3f}\n"
            coin_prompt += f"3‑Period ATR: {latest_longterm['ATRr_3']:.3f} vs. 14‑Period ATR: {latest_longterm['ATRr_14']:.3f}\n"
            coin_prompt += f"Current Volume: {latest_longterm['volume']} vs. Average Volume: {longterm_df['volume'].mean():.3f}\n"
            coin_prompt += f"MACD indicators: {[round(x, 3) for x in longterm_series['MACDh_12_26_9'].tolist()]}\n"
            coin_prompt += f"RSI indicators (14‑Period): {[round(x, 3) for x in longterm_series['RSI_14'].tolist()]}\n\n"
            
            master_prompt += coin_prompt
        except Exception as e:
            print(f"An error occurred processing {coin}: {e}")
    return master_prompt

def get_account_state(exchange_instance):
    """
    Fetches the current account balance and open positions from the exchange
    and formats them into a string for the LLM prompt.
    """
    print("Fetching account state...")
    try:
        balance = exchange_instance.fetch_balance()
        positions = exchange_instance.fetch_positions()

        total_balance = balance['total']['USDT']
        available_cash = balance['free']['USDT']

        open_positions = [p for p in positions if p.get('entryPrice') is not None and float(p.get('contracts', 0)) != 0]

        positions_str_list = []
        for p in open_positions:
            info = p.get('info', {})
            position_details = (
                f"{{'symbol': '{info.get('symbol', 'N/A')}', "
                f"'quantity': {info.get('positionAmt', 'N/A')}, "
                f"'entry_price': {info.get('entryPrice', 'N/A')}, "
                f"'current_price': {info.get('markPrice', 'N/A')}, "
                f"'liquidation_price': {info.get('liquidationPrice', 'N/A')}, "
                f"'unrealized_pnl': {info.get('unRealizedProfit', 'N/A')}, "
                f"'leverage': {info.get('leverage', 'N/A')}}}"
            )
            positions_str_list.append(position_details)
        
        positions_str = " ".join(positions_str_list) if positions_str_list else "{}"

        starting_capital = 5000 
        pnl_percent = ((total_balance - starting_capital) / starting_capital) * 100

        account_prompt = "HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE\n"
        account_prompt += f"Current Total Return (percent): {pnl_percent:.2f}%\n"
        account_prompt += f"Available Cash: {available_cash:.2f}\n"
        account_prompt += f"Current Account Value: {total_balance:.2f}\n"
        account_prompt += f"Current live positions & performance: {positions_str}\n"
        account_prompt += "Sharpe Ratio: 0.0\n"
        
        return account_prompt

    except Exception as e:
        print(f"Error fetching account state: {e}")
        return "ERROR: Could not fetch account information.\n"

def query_llm(user_prompt, system_prompt):
    """
    Sends the system prompt and user prompt to the DeepSeek API 
    and gets a trading decision back.
    """
    print("Querying DeepSeek for decision...")
    try:
        url = "https://api.deepseek.com/chat/completions"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # The payload correctly separates the two prompts into their roles
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt}, # The instructions from your .md file
                {"role": "user", "content": user_prompt}      # The live data from the exchange
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        response_json = response.json()
        decision_json = response_json['choices'][0]['message']['content']
        
        print("DeepSeek Decision Received.")
        return decision_json

    except requests.exceptions.RequestException as e:
        print(f"Error making request to DeepSeek API: {e}")
        if 'response' in locals() and response.text:
            print(f"Response Body: {response.text}")
        return '{"action": "hold", "reason": "DeepSeek API request failed."}'
    except Exception as e:
        print(f"An unexpected error occurred in query_llm: {e}")
        return '{"action": "hold", "reason": "An unexpected error occurred."}'


# --- Main Bot Loop ---
if __name__ == "__main__":
    # --- TEMPORARY TEST ---
    # You can now test the whole chain, up to the LLM query
    
    # Get the data
    account_state_prompt = get_account_state(exchange)
    market_data_prompt = get_market_data(exchange)

    # Combine into the user prompt
    user_prompt = f"{market_data_prompt}\n\n{account_state_prompt}"

    # Query the LLM, passing in the prompt we loaded from the file
    decision = query_llm(user_prompt, SYSTEM_PROMPT)

    print("\n--- LLM DECISION ---")
    print(decision)
    
    # --- REAL BOT LOOP (currently commented out) ---
    """
    while True:
        try:
            account_state_prompt = get_account_state(exchange) 
            market_data_prompt = get_market_data(exchange)
            system_prompt = "You are a world-class quantitative trading agent..."
            full_prompt = f"{system_prompt}\\n\\n{market_data_prompt}\\n\\n{account_state_prompt}"
            decision_json = query_llm(full_prompt)
            execute_trade(exchange, decision_json)
            print("Cycle complete. Sleeping for 3 minutes...")
            time.sleep(180)
        except Exception as e:
            print(f"CRITICAL ERROR in main loop: {e}")
            time.sleep(60)
    """