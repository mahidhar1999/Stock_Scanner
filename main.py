import pandas as pd
import yfinance as yf
import numpy as np
import time

# ---------------------------
# LOAD ALL SYMBOLS
# ---------------------------
print("Loading all NIFTY stocks from CSV...")
df = pd.read_csv("stocks.csv")

symbols = (df["SYMBOL"] + ".NS").tolist()
total_symbols = len(symbols)

print(f"Total symbols: {total_symbols}")

# ---------------------------
# DOWNLOAD NIFTY
# ---------------------------
print("\nDownloading NIFTY Index (^NSEI)...")
nifty = yf.download("^NSEI", period="300d", auto_adjust=True)

nifty_close_cols = [c for c in nifty.columns if "Close" in c]
nifty_close = nifty[nifty_close_cols[0]]

# ---------------------------
# PROCESS IN BATCHES OF 50
# ---------------------------
batch_size = 500
results = []

def process_batch(batch_symbols):
    print(f"\nDownloading batch: {batch_symbols}")

    data = yf.download(
        batch_symbols,
        period="300d",
        group_by="ticker",
        auto_adjust=True,
        threads=True
    )

    for symbol in batch_symbols:
        print(f"Processing {symbol}...")
        try:
            stock = data[symbol].copy()
            if stock.empty:
                print("  → No data for symbol")
                continue

            # Get Close column
            close_cols = [c for c in stock.columns if "Close" in c]
            close_series = stock[close_cols[0]]
            stock["CLOSE_FIX"] = close_series

            # EMAs
            stock["EMA5"] = stock["CLOSE_FIX"].ewm(span=5).mean()
            stock["EMA20"] = stock["CLOSE_FIX"].ewm(span=20).mean()
            stock["EMA50"] = stock["CLOSE_FIX"].ewm(span=50).mean()
            stock["EMA200"] = stock["CLOSE_FIX"].ewm(span=200).mean()

            # Align NIFTY
            nifty_aligned = nifty_close.asof(stock.index)

            close_arr = stock["CLOSE_FIX"].to_numpy()
            nifty_arr = nifty_aligned.to_numpy()

            # Shift arrays
            shift = 65
            close_shift = np.concatenate([np.full(shift, np.nan), close_arr[:-shift]])
            nifty_shift = np.concatenate([np.full(shift, np.nan), nifty_arr[:-shift]])

            # RS
            rs_arr = (close_arr / close_shift) / (nifty_arr / nifty_shift) - 1
            stock["RS"] = pd.Series(rs_arr, index=stock.index)

            last = stock.iloc[-1]

            results.append({
                "Symbol": symbol.replace(".NS", ""),
                "Close": last["CLOSE_FIX"],
                "RS": last["RS"],
                "EMA5": last["EMA5"],
                "EMA20": last["EMA20"],
                "EMA50": last["EMA50"],
                "EMA200": last["EMA200"],
            })

        except Exception as e:
            print("  → Error:", e)


# ---------------------------
# RUN BATCHES
# ---------------------------
for i in range(0, total_symbols, batch_size):
    batch = symbols[i:i+batch_size]
    process_batch(batch)
    time.sleep(2)   # prevent rate-limit


# ---------------------------
# BUILD RESULT CSV
# ---------------------------
result_df = pd.DataFrame(results)
print("\nFINAL RESULT:")
print(result_df)

filtered = result_df[
    (result_df["RS"] > 0) &
    (result_df["EMA5"] > result_df["EMA20"]) &
    (result_df["EMA20"] > result_df["EMA50"]) &
    (result_df["EMA50"] > result_df["EMA200"])
]

print("\nFILTERED STOCKS:")
print(filtered)

result_df.to_csv("all_stocks_raw.csv", index=False)
filtered.to_csv("all_stocks_filtered.csv", index=False)

print("\nSaved all_stocks_raw.csv & all_stocks_filtered.csv")
