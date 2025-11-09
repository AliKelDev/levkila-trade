# Prompt Snapshot
Generated at 2025-11-05T11:12:59.602610+00:00 UTC

## System Prompt

```text
PRIMARY DIRECTIVES:
Follow every rule below exactly as written.
Rely solely on the numbers provided in the prompt.
Focus on risk management and precise execution—no extra commentary.
RULES OF ENGAGEMENT:
Position Management: For each of the six assets (BTC, ETH, SOL, BNB, DOGE, XRP), you can only have one position at a time.
Action Space:
If you have no position in an asset, your only allowed actions are buy_to_enter or sell_to_enter.
If you have an existing position, your only allowed actions are hold or close_position.
Pyramiding is forbidden. You cannot add to an existing position.
The Exit Plan is Law: When you enter a position, you define an exit_plan containing a profit_target, stop_loss, and an invalidation_condition.
The profit target and stop loss are managed automatically by the system.
Your primary responsibility is to monitor the invalidation_condition. If this condition is met, you MUST issue a close_position action. Otherwise, you hold.
Think First, Act Second: Keep your reasoning short and directly focused on the current data—do not restate rules or prompt text. Within that reasoning you MUST include exactly one block formatted as `<FINAL_JSON>{ ... }</FINAL_JSON>` containing the decision JSON described below (no extra text inside the block). Your final assistant message must contain only that JSON block; do not add summaries or prose.
MANDATORY OUTPUT FORMAT:
• Reasoning example:
  ...concise analysis...
  `<FINAL_JSON>{
    "BTC": { "trade_signal_args": { ... } }
  }</FINAL_JSON>`
• Assistant message: the exact same JSON, e.g. `{ "BTC": { ... } }`
If you are unable to produce a valid decision, place `{ "error": "explanation" }` inside `<FINAL_JSON>`.
The JSON object inside `<FINAL_JSON>` contains a key for each asset you are acting upon. The value for each key is a JSON object specifying your decision.
The JSON object will contain a key for each asset you are acting upon. The value for each key will be a JSON object specifying your decision.
For hold actions:
code
JSON
{
    "BTC": {
        "trade_signal_args": {
            "signal": "hold",
            ... [copy all existing position parameters precisely from the user prompt]
        }
    }
}
For close_position actions:
code
JSON
{
    "ETH": {
        "trade_signal_args": {
            "signal": "close_position",
            "coin": "ETH",
            "quantity": <current position size>,
            "justification": "<Brief reason for closing, likely because the invalidation condition was met.>"
        }
    }
}
For buy_to_enter or sell_to_enter actions:
code
JSON
{
    "SOL": {
        "trade_signal_args": {
            "signal": "buy_to_enter",
            "coin": "SOL",
            "confidence": <float, 0.0-1.0>,
            "leverage": <int>,
            "risk_usd": <float>,
            "justification": "<Brief reasoning based on technical indicators.>",
            "exit_plan": {
                "profit_target": <float>,
                "stop_loss": <float>,
                "invalidation_condition": "<A clear, objective rule for when your thesis is wrong.>"
            }
        }
    }
}
"
```

## User Prompt

```text
It has been 6333 minutes since you started trading. The current time is 2025-11-05 11:12:59.602610+00:00 and you've been invoked 615 times. Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, positions, etc.

ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST

CURRENT MARKET STATE FOR ALL COINS
ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST

ALL BTC DATA
current_price = 101542.8, current_ema20 = 101570.692, current_macd = 5.157, current_rsi (7 period) = 52.018

Open Interest: Latest: 83409.38700 Average: 83413.90480
Funding Rate: 3.822e-05
Intraday series (by minute, oldest → latest):
Mid prices: [101557.4, 101400.0, 101395.2, 101340.0, 101359.8, 101422.4, 101521.6, 101511.1, 101560.8, 101542.8]
EMA indicators (20‑period): [101720.604, 101690.071, 101661.988, 101631.322, 101605.463, 101588.028, 101581.702, 101574.978, 101573.628, 101570.692]
MACD indicators: [-20.121, -31.989, -37.775, -42.575, -41.622, -34.287, -20.922, -11.432, -1.0, 5.157]
RSI indicators (7‑Period): [24.807, 15.712, 15.51, 13.225, 18.264, 32.682, 49.237, 47.785, 55.094, 52.018]

Longer‑term context (4‑hour timeframe):
20‑Period EMA: 105147.463 vs. 50‑Period EMA: 107712.932
3‑Period ATR: 2015.847 vs. 14‑Period ATR: 2129.185
Current Volume: 185564.215 vs. Average Volume: 142713.087
MACD indicators: [-334.401, -273.136, -385.917, -423.53, -503.985, -687.096, -694.362, -608.613, -512.217, -440.908]
RSI indicators (14‑Period): [36.398, 39.856, 30.999, 31.489, 27.859, 22.485, 27.821, 31.291, 31.149, 29.874]

ALL ETH DATA
current_price = 3290.61, current_ema20 = 3288.890, current_macd = 0.883, current_rsi (7 period) = 61.017

Open Interest: Latest: 1625964.59800 Average: 1627617.80760
Funding Rate: -4.38e-06
Intraday series (by minute, oldest → latest):
Mid prices: [3285.91, 3281.23, 3281.64, 3278.77, 3278.77, 3277.39, 3283.28, 3287.89, 3289.89, 3290.61]
EMA indicators (20‑period): [3295.637, 3294.265, 3293.063, 3291.702, 3290.47, 3289.224, 3288.658, 3288.585, 3288.709, 3288.89]
MACD indicators: [-0.305, -0.829, -1.057, -1.295, -1.341, -1.348, -0.865, -0.178, 0.433, 0.883]
RSI indicators (7‑Period): [29.855, 24.124, 25.584, 22.109, 22.109, 20.305, 43.336, 55.167, 59.452, 61.017]

Longer‑term context (4‑hour timeframe):
20‑Period EMA: 3541.600 vs. 50‑Period EMA: 3716.352
3‑Period ATR: 114.261 vs. 14‑Period ATR: 122.266
Current Volume: 993723.948 vs. Average Volume: 1280334.402
MACD indicators: [-25.158, -21.519, -27.398, -27.068, -26.079, -40.98, -43.73, -39.377, -33.863, -30.128]
RSI indicators (14‑Period): [29.26, 34.841, 26.706, 30.368, 29.41, 19.842, 25.342, 29.484, 29.352, 27.824]

ALL SOL DATA
current_price = 155.65, current_ema20 = 155.555, current_macd = 0.027, current_rsi (7 period) = 60.644

Open Interest: Latest: 7994384.28000 Average: 8013470.01300
Funding Rate: -3.9e-07
Intraday series (by minute, oldest → latest):
Mid prices: [155.47, 155.43, 155.0, 155.04, 154.78, 154.81, 155.0, 155.35, 155.53, 155.65]
EMA indicators (20‑period): [156.045, 155.987, 155.893, 155.812, 155.713, 155.627, 155.568, 155.547, 155.545, 155.555]
MACD indicators: [-0.112, -0.107, -0.125, -0.125, -0.134, -0.129, -0.104, -0.058, -0.013, 0.027]
RSI indicators (7‑Period): [29.88, 28.848, 20.127, 22.665, 18.265, 20.346, 32.961, 49.984, 56.597, 60.644]

Longer‑term context (4‑hour timeframe):
20‑Period EMA: 166.671 vs. 50‑Period EMA: 177.122
3‑Period ATR: 6.465 vs. 14‑Period ATR: 7.002
Current Volume: 197129.0 vs. Average Volume: 283803.840
MACD indicators: [-1.867, -1.704, -2.148, -1.975, -1.729, -1.869, -1.772, -1.448, -1.132, -0.858]
RSI indicators (14‑Period): [24.911, 28.396, 21.112, 29.668, 29.505, 24.936, 25.808, 29.045, 28.736, 28.117]

ALL BNB DATA
current_price = 940.71, current_ema20 = 941.041, current_macd = 0.131, current_rsi (7 period) = 53.998

Open Interest: Latest: 539856.17000 Average: 539807.89700
Funding Rate: 0.0
Intraday series (by minute, oldest → latest):
Mid prices: [942.11, 940.54, 938.74, 938.75, 937.41, 937.61, 938.47, 940.38, 940.6, 940.71]
EMA indicators (20‑period): [943.456, 943.178, 942.756, 942.374, 941.901, 941.493, 941.205, 941.126, 941.076, 941.041]
MACD indicators: [-0.15, -0.196, -0.32, -0.37, -0.457, -0.463, -0.375, -0.165, 0.004, 0.131]
RSI indicators (7‑Period): [43.225, 33.568, 25.845, 25.956, 21.055, 23.568, 34.093, 51.424, 53.082, 53.998]

Longer‑term context (4‑hour timeframe):
20‑Period EMA: 990.647 vs. 50‑Period EMA: 1039.526
3‑Period ATR: 30.857 vs. 14‑Period ATR: 36.010
Current Volume: 2122752.03 vs. Average Volume: 3863085.277
MACD indicators: [-9.76, -8.523, -10.593, -10.268, -9.678, -10.631, -9.327, -6.919, -4.434, -3.212]
RSI indicators (14‑Period): [23.437, 28.53, 20.45, 26.118, 25.291, 21.248, 27.729, 32.26, 33.958, 31.542]

ALL XRP DATA
current_price = 2.2204, current_ema20 = 2.220, current_macd = 0.000, current_rsi (7 period) = 55.675

Open Interest: Latest: 190847167.90000 Average: 190727739.69000
Funding Rate: 3.764e-05
Intraday series (by minute, oldest → latest):
Mid prices: [2.2191, 2.2122, 2.2117, 2.21, 2.2105, 2.2119, 2.2209, 2.2202, 2.2196, 2.2204]
EMA indicators (20‑period): [2.227, 2.225, 2.224, 2.223, 2.221, 2.221, 2.221, 2.221, 2.22, 2.22]
MACD indicators: [-0.001, -0.002, -0.002, -0.002, -0.002, -0.002, -0.001, -0.0, 0.0, 0.0]
RSI indicators (7‑Period): [30.759, 21.383, 20.846, 18.957, 21.401, 28.45, 57.222, 55.208, 53.33, 55.675]

Longer‑term context (4‑hour timeframe):
20‑Period EMA: 2.325 vs. 50‑Period EMA: 2.414
3‑Period ATR: 0.077 vs. 14‑Period ATR: 0.071
Current Volume: 1386687891.4 vs. Average Volume: 1971130197.446
MACD indicators: [-0.019, -0.016, -0.02, -0.019, -0.017, -0.021, -0.02, -0.016, -0.011, -0.008]
RSI indicators (14‑Period): [25.861, 35.25, 26.565, 32.469, 32.719, 25.574, 30.66, 35.043, 36.317, 34.371]

ALL DOGE DATA
current_price = 0.16233, current_ema20 = 0.162, current_macd = 0.000, current_rsi (7 period) = 56.547

Open Interest: Latest: 1585028094.00000 Average: 1583739898.30000
Funding Rate: 6.455e-05
Intraday series (by minute, oldest → latest):
Mid prices: [0.16212, 0.16151, 0.16155, 0.16137, 0.16144, 0.16164, 0.1621, 0.16227, 0.16226, 0.16233]
EMA indicators (20‑period): [0.163, 0.163, 0.163, 0.163, 0.162, 0.162, 0.162, 0.162, 0.162, 0.162]
MACD indicators: [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0]
RSI indicators (7‑Period): [24.801, 17.364, 19.217, 17.193, 20.97, 31.4, 49.339, 54.473, 54.096, 56.547]

Longer‑term context (4‑hour timeframe):
20‑Period EMA: 0.170 vs. 50‑Period EMA: 0.179
3‑Period ATR: 0.007 vs. 14‑Period ATR: 0.006
Current Volume: 15399748979.0 vs. Average Volume: 23513379685.690
MACD indicators: [-0.002, -0.001, -0.002, -0.001, -0.001, -0.001, -0.001, -0.001, -0.0, -0.0]
RSI indicators (14‑Period): [23.649, 32.415, 25.109, 33.02, 31.251, 26.747, 33.988, 37.198, 38.127, 35.682]


HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE
Current Total Return (percent): -14.30%
Current Account Value: 4285.04872428
Current live positions & performance: {'symbol': 'SOL', 'quantity': 38.0, 'entry_price': 156.86, 'current_price': 155.75366286, 'liquidation_price': 44.54096958, 'unrealized_pnl': -42.04081132, 'leverage': 0, 'exit_plan': {'invalidation_condition': 'If 1-minute closing price falls below 155.0', 'profit_target': 163.0, 'stop_loss': 154.0}, 'confidence': 0.6, 'risk_usd': 100.0, 'sl_oid': '1213237557', 'tp_oid': '1213237845', 'wait_for_fill': False, 'entry_oid': '1213237433', 'notional_usd': 5951.94}
Sharpe Ratio: 0.0
```
